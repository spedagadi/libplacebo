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
 *   78 floats (features[0..77]) in native float32 format
 *
 * Implementation note:
 *   Features are extracted via pl_shader_detect_peak, which runs inside
 *   pl_render_image as part of the normal rendering pipeline. This matches
 *   the exact execution path used by mpv/vo_gpu_next. After rendering,
 *   pl_renderer_get_ml_features reads the accumulated stats from the GPU
 *   peak detection buffer — no separate downscale pass.
 *
 * Usage:
 *   pl_extract_features_daemon  (reads from stdin, writes to stdout)
 *
 * Python integration:
 *   extractor = SustainedFeatureExtractor("pl_extract_features_daemon.exe")
 *   extractor.start()
 *   features = extractor.extract_frame("video.mkv", 123.45, target_nits=100.0)
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
#include <libplacebo/renderer.h>
#include <libplacebo/utils/libav.h>

#define PL_LIBAV_IMPLEMENTATION 1
#include <libplacebo/utils/libav.h>

#define MAX_INPUT_LINE 4096
#define ML_FEATURE_DIM 78

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
    req->target_nits = 100.0f;

    const char *mkv_start = strstr(line, "\"mkv_path\"");
    const char *pts_start = strstr(line, "\"pts\"");
    const char *nits_start = strstr(line, "\"target_nits\"");

    if (!mkv_start || !pts_start) {
        fprintf(stderr, "ERROR: Invalid JSON - missing mkv_path or pts\n");
        return false;
    }

    const char *path_value = strchr(mkv_start, ':');
    if (path_value) {
        path_value = strchr(path_value, '"');
        if (path_value) {
            path_value++;
            const char *path_end = strchr(path_value, '"');
            if (path_end) {
                size_t len = path_end - path_value;
                if (len >= sizeof(req->mkv_path)) len = sizeof(req->mkv_path) - 1;
                strncpy(req->mkv_path, path_value, len);
                req->mkv_path[len] = '\0';
            }
        }
    }

    const char *pts_value = strchr(pts_start, ':');
    if (pts_value)
        req->pts = atof(pts_value + 1);

    if (nits_start) {
        const char *nits_value = strchr(nits_start, ':');
        if (nits_value)
            req->target_nits = atof(nits_value + 1);
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
    double last_decoded_pts;
} VideoState;

static VideoState *open_video(const char *path)
{
    VideoState *vs = calloc(1, sizeof(VideoState));
    if (!vs) return NULL;

#ifdef _WIN32
    char native_path[sizeof(vs->path)];
    const char *open_path = path;
    if (path[0] == '/' && path[1] && path[2] == '/') {
        snprintf(native_path, sizeof(native_path), "%c:%s", path[1], path + 2);
        open_path = native_path;
    }
#endif

    strncpy(vs->path, path, sizeof(vs->path) - 1);
    vs->last_decoded_pts = -1.0;

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
    AVCodecContext  *dec = vs->dec;
    AVStream        *st  = vs->st;
    int vstream = vs->vstream;

    bool sequential = (vs->last_decoded_pts >= 0.0 &&
                       target_pts >= vs->last_decoded_pts &&
                       (target_pts - vs->last_decoded_pts) < 1.0);

    if (!sequential) {
        int64_t seek_ts = (int64_t)(target_pts / av_q2d(st->time_base));
        av_seek_frame(fmt, vstream, seek_ts, AVSEEK_FLAG_BACKWARD);
        avcodec_flush_buffers(dec);
    }

    AVPacket *pkt   = av_packet_alloc();
    AVFrame  *frame = av_frame_alloc();
    AVFrame  *best  = NULL;
    double    best_diff = 1e9;

    for (int attempts = 0; attempts < 512; attempts++) {
        int ret = av_read_frame(fmt, pkt);
        if (ret < 0) break;
        if (pkt->stream_index != vstream) { av_packet_unref(pkt); continue; }

        avcodec_send_packet(dec, pkt);
        av_packet_unref(pkt);

        while (avcodec_receive_frame(dec, frame) == 0) {
            double pts  = frame->pts * av_q2d(st->time_base);
            double diff = fabs(pts - target_pts);
            if (diff < best_diff) {
                best_diff = diff;
                if (best) av_frame_free(&best);
                best = av_frame_clone(frame);
                if (diff < 0.001) goto found;
            }
            av_frame_unref(frame);
        }
    }

found:
    av_packet_free(&pkt);
    av_frame_free(&frame);
    if (best)
        vs->last_decoded_pts = best->pts * av_q2d(st->time_base);
    return best;
}

/* -------------------------------------------------------------------------
 * Persistent render target — recreated when resolution changes
 * ---------------------------------------------------------------------- */
typedef struct {
    pl_tex  tex;
    int     w, h;
} RenderTarget;

static bool ensure_render_target(pl_gpu gpu, RenderTarget *rt, int w, int h)
{
    if (rt->tex && rt->w == w && rt->h == h)
        return true;

    pl_tex_destroy(gpu, &rt->tex);
    rt->w = rt->h = 0;

    // RGBA16F — suitable for HDR content, write-only from renderer
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 16,
                             PL_FMT_CAP_RENDERABLE);
    if (!fmt) {
        fprintf(stderr, "ERROR: No RGBA16F renderable format\n");
        return false;
    }

    rt->tex = pl_tex_create(gpu, pl_tex_params(
        .w = w, .h = h, .format = fmt,
        .renderable = true, .blit_dst = true));
    if (!rt->tex) {
        fprintf(stderr, "ERROR: Failed to create render target %dx%d\n", w, h);
        return false;
    }

    rt->w = w;
    rt->h = h;
    return true;
}

/* -------------------------------------------------------------------------
 * Process single frame request
 *
 * Renders via pl_render_image (which fires pl_shader_detect_peak), then
 * reads the accumulated ML features via pl_renderer_get_ml_features.
 * This mirrors the exact path used by mpv/vo_gpu_next on every frame.
 * ---------------------------------------------------------------------- */
bool g_no_dovi = false;

static bool process_request(pl_gpu gpu, pl_renderer renderer,
                            RenderTarget *rt, VideoState *vs,
                            const FrameRequest *req, float *features_out)
{
    AVFrame *frame = decode_frame_at(vs, req->pts);
    if (!frame) {
        fprintf(stderr, "ERROR: Failed to decode frame at PTS %.3f\n", req->pts);
        return false;
    }

    // Map AVFrame → GPU textures.
    // Always strip DV RPU metadata (map_dovi = false) so that hdr_update_peak
    // runs GPU histogram peak detection.  When map_dovi=true, the renderer sees
    // L1 avg_pq_y from the RPU and skips GPU peak detection entirely, which
    // would leave the peak buffer empty and pl_renderer_get_ml_features failing.
    struct pl_frame src = {0};
    pl_tex tex[4] = {0};
    struct pl_avframe_params avp = {
        .frame    = frame,
        .tex      = tex,
        .map_dovi = false,
    };
    if (!pl_map_avframe_ex(gpu, &src, &avp)) {
        fprintf(stderr, "ERROR: Failed to map AVFrame to GPU\n");
        av_frame_free(&frame);
        return false;
    }

    // Ensure off-screen render target matches source resolution.
    // Rendering at source resolution means pl_shader_detect_peak sees the
    // upscaling path → runs at SOURCE resolution (no downscale yet), which
    // matches mpv's behaviour for 4K HDR content on a 4K display.
    int w = frame->width, h = frame->height;
    if (!ensure_render_target(gpu, rt, w, h)) {
        pl_unmap_avframe(gpu, &src);
        for (int i = 0; i < 4; i++) pl_tex_destroy(gpu, &tex[i]);
        av_frame_free(&frame);
        return false;
    }

    // Build target pl_frame pointing at the off-screen texture
    struct pl_frame dst = {0};
    dst.num_planes = 1;
    dst.planes[0].texture = rt->tex;
    dst.planes[0].components = 4;
    dst.planes[0].component_mapping[0] = PL_CHANNEL_R;
    dst.planes[0].component_mapping[1] = PL_CHANNEL_G;
    dst.planes[0].component_mapping[2] = PL_CHANNEL_B;
    dst.planes[0].component_mapping[3] = PL_CHANNEL_A;
    dst.repr  = pl_color_repr_hdtv;
    dst.color = pl_color_space_srgb;
    dst.crop  = (struct pl_rect2df){ 0, 0, w, h };

    // Peak detect params.
    // allow_delayed = true: avoids requiring pass->fbofmt[4] (fp16 FBO) which
    // may not be available in a minimal off-screen render setup.  Instead we
    // call pl_gpu_finish() after rendering to ensure GPU writes are complete.
    static const struct pl_peak_detect_params pd = {
        .smoothing_period     = 1.0f,
        .scene_threshold_low  = 0.0f,
        .scene_threshold_high = 0.0f,
        .percentile           = 99.995f,
        .allow_delayed        = true,
    };

    struct pl_render_params rp = pl_render_default_params;
    rp.peak_detect_params = &pd;

    bool rendered = pl_render_image(renderer, &src, &dst, &rp);

    bool ok = false;
    if (rendered) {
        // Wait for all queued GPU commands to complete so the peak detection
        // SSBO writes are visible before we read them back.
        pl_gpu_finish(gpu);
        ok = pl_renderer_get_ml_features(renderer, req->target_nits, features_out);
        if (!ok)
            fprintf(stderr, "WARN: Peak detection buffer not ready after gpu_finish\n");
    } else {
        fprintf(stderr, "ERROR: pl_render_image failed\n");
    }

    pl_unmap_avframe(gpu, &src);
    for (int i = 0; i < 4; i++) pl_tex_destroy(gpu, &tex[i]);
    av_frame_free(&frame);
    return ok;
}

/* -------------------------------------------------------------------------
 * Main daemon loop
 * ---------------------------------------------------------------------- */
static void usage(const char *argv0)
{
    fprintf(stderr,
            "Usage: %s [--no-dovi]\n"
            "  --no-dovi   Skip Dolby Vision RPU processing\n"
            "Reads JSON requests from stdin, writes 78 float32 feature vectors to stdout.\n",
            argv0);
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-dovi") == 0)
            g_no_dovi = true;
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]); return 0;
        } else {
            fprintf(stderr, "ERROR: Unknown argument: %s\n", argv[i]);
            usage(argv[0]); return 1;
        }
    }

#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    av_log_set_level(AV_LOG_ERROR);

    // Persistent GPU context + renderer (created once, reused for all requests)
    pl_log log = pl_log_create(PL_API_VER, pl_log_params(
        .log_cb    = pl_log_simple,
        .log_priv  = stderr,
        .log_level = PL_LOG_WARN,
    ));

    pl_d3d11 d3d11 = pl_d3d11_create(log, pl_d3d11_params(.allow_software = true));
    if (!d3d11) {
        fprintf(stderr, "FATAL: Failed to create D3D11 context\n");
        pl_log_destroy(&log);
        return 1;
    }
    pl_gpu gpu = d3d11->gpu;

    pl_renderer renderer = pl_renderer_create(log, gpu);
    if (!renderer) {
        fprintf(stderr, "FATAL: Failed to create renderer\n");
        pl_d3d11_destroy(&d3d11);
        pl_log_destroy(&log);
        return 1;
    }

    RenderTarget rt = {0};

    fprintf(stderr, "Daemon ready: GPU context + renderer initialized\n");

    char line[MAX_INPUT_LINE];
    int request_count = 0;
    VideoState *video_state = NULL;

    while (fgets(line, sizeof(line), stdin) != NULL) {
        request_count++;

        FrameRequest req;
        if (!parse_json_request(line, &req)) {
            fprintf(stderr, "ERROR: Failed to parse request #%d\n", request_count);
            float zeros[ML_FEATURE_DIM] = {0};
            fwrite(zeros, sizeof(float), ML_FEATURE_DIM, stdout);
            fflush(stdout);
            continue;
        }

        if (!video_state || strcmp(video_state->path, req.mkv_path) != 0) {
            if (video_state) close_video(video_state);
            video_state = open_video(req.mkv_path);
            if (!video_state) {
                fprintf(stderr, "ERROR: Failed to open video: %s\n", req.mkv_path);
                float zeros[ML_FEATURE_DIM] = {0};
                fwrite(zeros, sizeof(float), ML_FEATURE_DIM, stdout);
                fflush(stdout);
                continue;
            }
            fprintf(stderr, "Opened video: %s\n", req.mkv_path);
        }

        float features[ML_FEATURE_DIM];
        bool ok = process_request(gpu, renderer, &rt, video_state, &req, features);
        if (!ok) {
            fprintf(stderr, "ERROR: Feature extraction failed for %s @ %.3f\n",
                    req.mkv_path, req.pts);
            memset(features, 0, sizeof(features));
        }

        size_t written = fwrite(features, sizeof(float), ML_FEATURE_DIM, stdout);
        fflush(stdout);
        if (written != ML_FEATURE_DIM)
            fprintf(stderr, "ERROR: Short write (%zu/%d floats)\n", written, ML_FEATURE_DIM);
    }

    fprintf(stderr, "Daemon shutting down after %d requests\n", request_count);

    if (video_state) close_video(video_state);
    pl_tex_destroy(gpu, &rt.tex);
    pl_renderer_destroy(&renderer);
    pl_d3d11_destroy(&d3d11);
    pl_log_destroy(&log);
    return 0;
}
