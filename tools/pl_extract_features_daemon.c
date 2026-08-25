/*
 * pl_extract_features_daemon — Persistent ML feature extractor daemon
 *
 * Accepts JSON commands via stdin, extracts features with persistent GPU context,
 * returns binary results via stdout. Eliminates process spawning overhead.
 *
 * Input format (one JSON object per line):
 *   {"mkv_path": "path/to/video.mkv", "pts": 123.45, "target_nits": 100.0}
 *
 * Output format (binary):
 *   78 floats (77 features + target_nits) in native float32 format
 *
 * Usage:
 *   pl_extract_features_daemon  (reads from stdin, writes to stdout)
 *
 * Python integration:
 *   extractor = SustainedFeatureExtractor("pl_extract_features_daemon.exe")
 *   extractor.start()
 *   features = extractor.extract_frame("video.mkv", 123.45, target_nits=100.0)
 *
 * Performance:
 *   - Single GPU context initialization (persistent)
 *   - No process spawning overhead
 *   - ~0.05-0.15s per frame (vs ~3.9s with single-shot binary)
 *
 * License: CC0 / Public Domain
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>

#include <libplacebo/log.h>
#include <libplacebo/gpu.h>
#include <libplacebo/d3d11.h>
#include <libplacebo/utils/libav.h>
#include <libplacebo/ml_features.h>

#define PL_LIBAV_IMPLEMENTATION 1
#include <libplacebo/utils/libav.h>

#define MAX_INPUT_LINE 4096

static const char *g_debug_luma_path = NULL;

static void usage(const char *argv0)
{
    fprintf(stderr,
            "Usage: %s [--no-dovi] [--debug-luma <file>]\n"
            "  --no-dovi                 Skip Dolby Vision RPU processing\n"
            "  --debug-luma <file>      Write the latest 256x144 float32 luma frame\n",
            argv0);
}

/* -------------------------------------------------------------------------
 * Request structure (parsed from JSON stdin)
 * ---------------------------------------------------------------------- */
typedef struct {
    char   mkv_path[2048];
    double pts;
    float  target_nits;
} FrameRequest;

/* -------------------------------------------------------------------------
 * Simple JSON parser (handles our specific schema only)
 * ---------------------------------------------------------------------- */
static bool parse_json_request(const char *line, FrameRequest *req)
{
    memset(req, 0, sizeof(*req));
    req->target_nits = 100.0f;  // Default

    // Very simple JSON parser - looks for our known fields
    // Format: {"mkv_path": "...", "pts": 123.45, "target_nits": 100.0}

    const char *mkv_start = strstr(line, "\"mkv_path\"");
    const char *pts_start = strstr(line, "\"pts\"");
    const char *nits_start = strstr(line, "\"target_nits\"");

    if (!mkv_start || !pts_start) {
        fprintf(stderr, "ERROR: Invalid JSON - missing mkv_path or pts\n");
        return false;
    }

    // Extract mkv_path string
    const char *path_value = strchr(mkv_start, ':');
    if (path_value) {
        path_value = strchr(path_value, '"');
        if (path_value) {
            path_value++; // Skip opening quote
            const char *path_end = strchr(path_value, '"');
            if (path_end) {
                size_t len = path_end - path_value;
                if (len >= sizeof(req->mkv_path)) len = sizeof(req->mkv_path) - 1;
                strncpy(req->mkv_path, path_value, len);
                req->mkv_path[len] = '\0';
            }
        }
    }

    // Extract pts float
    const char *pts_value = strchr(pts_start, ':');
    if (pts_value) {
        req->pts = atof(pts_value + 1);
    }

    // Extract target_nits (optional)
    if (nits_start) {
        const char *nits_value = strchr(nits_start, ':');
        if (nits_value) {
            req->target_nits = atof(nits_value + 1);
        }
    }

    if (req->mkv_path[0] == '\0') {
        fprintf(stderr, "ERROR: Failed to parse mkv_path from JSON\n");
        return false;
    }

    return true;
}

/* -------------------------------------------------------------------------
 * Persistent video state (reused across requests for same file)
 * ---------------------------------------------------------------------- */
typedef struct {
    char path[2048];
    AVFormatContext *fmt;
    AVCodecContext *dec;
    AVStream *st;
    int vstream;
    double last_decoded_pts;  // Track last position for sequential seek optimization
} VideoState;

static VideoState *open_video(const char *path)
{
    VideoState *vs = calloc(1, sizeof(VideoState));
    if (!vs) return NULL;

#ifdef _WIN32
    char native_path[sizeof(vs->path)];
    const char *open_path = path;
    if (path[0] == '/' && path[1] && path[2] == '/') {
        snprintf(native_path, sizeof(native_path), "%c:%s",
                 path[1], path + 2);
        open_path = native_path;
    }
#endif

    strncpy(vs->path, path, sizeof(vs->path) - 1);
    vs->last_decoded_pts = -1.0;  // Initialize seek tracker

    if (avformat_open_input(&vs->fmt,
#ifdef _WIN32
                            open_path,
#else
                            path,
#endif
                            NULL, NULL) < 0) {
        fprintf(stderr, "ERROR: Cannot open: %s\n", path);
        free(vs);
        return NULL;
    }
    avformat_find_stream_info(vs->fmt, NULL);

    vs->vstream = av_find_best_stream(vs->fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (vs->vstream < 0) {
        fprintf(stderr, "ERROR: No video stream in %s\n", path);
        avformat_close_input(&vs->fmt);
        free(vs);
        return NULL;
    }

    vs->st = vs->fmt->streams[vs->vstream];
    const AVCodec *codec = avcodec_find_decoder(vs->st->codecpar->codec_id);
    vs->dec = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(vs->dec, vs->st->codecpar);

    if (avcodec_open2(vs->dec, codec, NULL) < 0) {
        fprintf(stderr, "ERROR: Cannot open decoder for %s\n", path);
        avcodec_free_context(&vs->dec);
        avformat_close_input(&vs->fmt);
        free(vs);
        return NULL;
    }

    return vs;
}

static void close_video(VideoState *vs)
{
    if (!vs) return;
    avcodec_free_context(&vs->dec);
    avformat_close_input(&vs->fmt);
    free(vs);
}

/* -------------------------------------------------------------------------
 * FFmpeg: decode one frame near the given PTS
 * ---------------------------------------------------------------------- */
static AVFrame *decode_frame_at(VideoState *vs, double target_pts)
{
    AVFormatContext *fmt = vs->fmt;
    AVCodecContext *dec = vs->dec;
    AVStream *st = vs->st;
    int vstream = vs->vstream;

    // Sequential seek optimization: skip seeking if target is close to last position
    bool sequential = (vs->last_decoded_pts >= 0.0 &&
                       target_pts >= vs->last_decoded_pts &&
                       (target_pts - vs->last_decoded_pts) < 1.0);

    if (!sequential) {
        // Seek to just before target PTS, using stream-specific index (not -1)
        int64_t seek_ts = (int64_t)(target_pts / av_q2d(st->time_base));
        av_seek_frame(fmt, vstream, seek_ts, AVSEEK_FLAG_BACKWARD);
        avcodec_flush_buffers(dec);
    }

    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    AVFrame *best = NULL;
    double best_diff = 1e9;

    for (int attempts = 0; attempts < 512; attempts++) {
        int ret = av_read_frame(fmt, pkt);
        if (ret < 0) break;
        if (pkt->stream_index != vstream) {
            av_packet_unref(pkt);
            continue;
        }

        avcodec_send_packet(dec, pkt);
        av_packet_unref(pkt);

        while (avcodec_receive_frame(dec, frame) == 0) {
            double pts = frame->pts * av_q2d(st->time_base);
            double diff = fabs(pts - target_pts);

            if (diff < best_diff) {
                best_diff = diff;
                if (best) av_frame_free(&best);  // CRITICAL: Free old frame before overwriting
                best = av_frame_clone(frame);

                // Accept if within 1ms (frame-exact at any reasonable FPS)
                // At 120fps, 1 frame = 8.3ms, so 1ms ensures exact frame match
                if (diff < 0.001)
                    goto found;
            }
            av_frame_unref(frame);
        }
    }

found:
    av_packet_free(&pkt);
    av_frame_free(&frame);

    // Update position tracker for sequential optimization
    if (best) {
        vs->last_decoded_pts = best->pts * av_q2d(st->time_base);
    }

    return best;
}

/* -------------------------------------------------------------------------
 * Process single frame request
 * ---------------------------------------------------------------------- */
static bool process_request(pl_gpu gpu, VideoState *vs, const FrameRequest *req, float *features_out)
{
    // Decode frame
    AVFrame *frame = decode_frame_at(vs, req->pts);
    if (!frame) {
        return false;
    }

    // Map AVFrame to GPU textures WITH Dolby Vision metadata
    struct pl_frame pl_frame;
    pl_tex tex[4] = {0};

    // Use pl_map_avframe_ex to properly handle Dolby Vision RPU metadata
    // map_dovi=false when --no-dovi flag is set (HDR10 calibration extraction mode)
    extern bool g_no_dovi;
    struct pl_avframe_params avparams = {
        .frame = frame,
        .tex = tex,
        .map_dovi = !g_no_dovi,
    };

    if (!pl_map_avframe_ex(gpu, &pl_frame, &avparams)) {
        fprintf(stderr, "ERROR: Failed to map frame to GPU\n");
        av_frame_free(&frame);
        return false;
    }

    // Extract features with persistent GPU context (use GPU, not CPU fallback)
    struct pl_ml_feature_params params = {
        .target_nits = req->target_nits,
        .downsample_width = 256,
        .downsample_height = 144,
        .force_cpu_fallback = false,  // Use native GPU paths for DV-corrected values
        .debug_luma_path = g_debug_luma_path,
    };

    bool ok = pl_extract_ml_features(gpu, &pl_frame, &params, features_out);

    // Clean up - Note: pl_unmap_avframe does NOT destroy textures (by design for reuse)
    pl_unmap_avframe(gpu, &pl_frame);
    for (int i = 0; i < 4; i++) {
        if (tex[i]) pl_tex_destroy(gpu, &tex[i]);
    }
    av_frame_free(&frame);

    return ok;
}

/* -------------------------------------------------------------------------
 * Main daemon loop
 * ---------------------------------------------------------------------- */
/* Global flag: when true, DV RPU polynomial is NOT applied (HDR10 calibration mode) */
bool g_no_dovi = false;

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-dovi") == 0)
            g_no_dovi = true;
        else if (strcmp(argv[i], "--debug-luma") == 0 && i + 1 < argc)
            g_debug_luma_path = argv[++i];
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "ERROR: Unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    // Binary mode on stdout only (stdin stays in text mode for fgets JSON parsing)
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    av_log_set_level(AV_LOG_ERROR);  // Suppress ffmpeg noise

    // Initialize libplacebo ONCE (persistent GPU context)
    pl_log log = pl_log_create(PL_API_VER, pl_log_params(
        .log_cb = pl_log_simple,
        .log_priv = stderr,
        .log_level = PL_LOG_WARN,
    ));

    pl_d3d11 d3d11 = pl_d3d11_create(log, pl_d3d11_params(
        .allow_software = true,
    ));

    if (!d3d11) {
        fprintf(stderr, "FATAL: Failed to create persistent D3D11 context\n");
        return 1;
    }
    pl_gpu gpu = d3d11->gpu;

    fprintf(stderr, "Daemon ready: GPU context initialized\n");

    // Main event loop: read JSON commands from stdin
    char line[MAX_INPUT_LINE];
    int request_count = 0;
    VideoState *video_state = NULL;  // Persistent video file handle

    while (fgets(line, sizeof(line), stdin) != NULL) {
        request_count++;

        // Parse JSON request
        FrameRequest req;
        if (!parse_json_request(line, &req)) {
            fprintf(stderr, "ERROR: Failed to parse request #%d\n", request_count);
            // Write error marker (all zeros)
            float zeros[PL_ML_FEATURE_DIM] = {0};
            fwrite(zeros, sizeof(float), PL_ML_FEATURE_DIM, stdout);
            fflush(stdout);
            continue;
        }

        // Check if video file changed - reopen if needed
        if (!video_state || strcmp(video_state->path, req.mkv_path) != 0) {
            if (video_state) {
                close_video(video_state);
            }
            video_state = open_video(req.mkv_path);
            if (!video_state) {
                fprintf(stderr, "ERROR: Failed to open video: %s\n", req.mkv_path);
                // Write error marker (all zeros)
                float zeros[PL_ML_FEATURE_DIM] = {0};
                fwrite(zeros, sizeof(float), PL_ML_FEATURE_DIM, stdout);
                fflush(stdout);
                continue;
            }
            fprintf(stderr, "Opened video: %s\n", req.mkv_path);
        }

        // Process frame with persistent GPU context AND persistent video handle
        float features[PL_ML_FEATURE_DIM];
        bool ok = process_request(gpu, video_state, &req, features);

        if (!ok) {
            fprintf(stderr, "ERROR: Failed to extract features for %s @ %.3f\n",
                    req.mkv_path, req.pts);
            // Write error marker (all zeros)
            memset(features, 0, sizeof(features));
        }

        // Write binary output to stdout
        size_t written = fwrite(features, sizeof(float), PL_ML_FEATURE_DIM, stdout);
        fflush(stdout);  // Critical: force immediate write

        if (written != PL_ML_FEATURE_DIM) {
            fprintf(stderr, "ERROR: Failed to write features (wrote %zu/%d)\n",
                    written, PL_ML_FEATURE_DIM);
        }
    }

    // Clean up when stdin closes
    fprintf(stderr, "Daemon shutting down: processed %d requests\n", request_count);

    if (video_state) {
        close_video(video_state);
    }

    pl_d3d11_destroy(&d3d11);
    pl_log_destroy(&log);

    return 0;
}
