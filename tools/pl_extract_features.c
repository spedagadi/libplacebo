/*
 * pl_extract_features — Standalone ML feature extractor
 *
 * Extracts 77-dimensional feature vector from a video frame at given PTS.
 * Output format: 78 floats in binary (77 features + target_nits)
 *
 * Usage:
 *   pl_extract_features --input video.mkv --pts 123.45 --target-nits 100 --output features.bin
 *   pl_extract_features --input video.mkv --pts 123.45 --target-nits 100  (stdout binary)
 *
 * Integration with Python training script:
 *   # In dv_metadata_extract.py
 *   subprocess.run(["pl_extract_features", "--input", mkv, "--pts", str(pts),
 *                   "--target-nits", "100", "--output", "features.bin"])
 *   features = np.fromfile("features.bin", dtype=np.float32, count=78)
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

/* -------------------------------------------------------------------------
 * Argument parsing
 * ---------------------------------------------------------------------- */
typedef struct {
    const char *input;
    const char *output;
    const char *debug_png;  // Optional: save downscaled luma as PNG
    double      pts;
    float       target_nits;
} Args;

static void usage(const char *argv0)
{
    fprintf(stderr,
        "Usage: %s --input <file> --pts <sec> [--target-nits <nits>] [--output <file>] [--debug-png <file>]\n"
        "  --input <file>        Input video file\n"
        "  --pts <sec>           Timestamp in seconds\n"
        "  --target-nits <nits>  Display target (default: 100.0)\n"
        "  --output <file>       Output binary file (default: stdout)\n"
        "  --debug-png <file>    Save downscaled luma as raw float32 (256x144)\n"
        "\n"
        "Output: 78 floats (77 features + target_nits) in native binary format\n"
        "\n"
        "Python integration example:\n"
        "  import numpy as np\n"
        "  features = np.fromfile('features.bin', dtype=np.float32, count=78)\n"
        "  luma = np.fromfile('luma.raw', dtype=np.float32).reshape(144, 256)\n",
        argv0);
}

static bool parse_args(int argc, char **argv, Args *a)
{
    memset(a, 0, sizeof(*a));
    a->target_nits = 100.0f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i+1 < argc)
            a->input = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i+1 < argc)
            a->output = argv[++i];
        else if (strcmp(argv[i], "--debug-png") == 0 && i+1 < argc)
            a->debug_png = argv[++i];
        else if (strcmp(argv[i], "--pts") == 0 && i+1 < argc)
            a->pts = atof(argv[++i]);
        else if (strcmp(argv[i], "--target-nits") == 0 && i+1 < argc)
            a->target_nits = atof(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
            return false;
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return false;
        }
    }

    if (!a->input) {
        fprintf(stderr, "Missing required argument: --input\n");
        return false;
    }

    return true;
}

/* -------------------------------------------------------------------------
 * FFmpeg: decode one frame near the given PTS
 * ---------------------------------------------------------------------- */
static AVFrame *decode_frame_at(const char *path, double target_pts)
{
    AVFormatContext *fmt = NULL;
    if (avformat_open_input(&fmt, path, NULL, NULL) < 0) {
        fprintf(stderr, "Cannot open: %s\n", path);
        return NULL;
    }
    avformat_find_stream_info(fmt, NULL);

    int vstream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (vstream < 0) {
        fprintf(stderr, "No video stream\n");
        avformat_close_input(&fmt);
        return NULL;
    }

    AVStream *st = fmt->streams[vstream];
    const AVCodec *codec = avcodec_find_decoder(st->codecpar->codec_id);
    AVCodecContext *dec = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(dec, st->codecpar);

    if (avcodec_open2(dec, codec, NULL) < 0) {
        fprintf(stderr, "Cannot open decoder\n");
        avcodec_free_context(&dec);
        avformat_close_input(&fmt);
        return NULL;
    }

    // Seek to just before target PTS
    int64_t seek_ts = (int64_t)(target_pts * AV_TIME_BASE);
    av_seek_frame(fmt, -1, seek_ts, AVSEEK_FLAG_BACKWARD);
    avcodec_flush_buffers(dec);

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
                av_frame_free(&best);
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
    avcodec_free_context(&dec);
    avformat_close_input(&fmt);

    if (best)
        fprintf(stderr, "Decoded frame at pts=%.3f (diff=%.3f sec)\n",
                best->pts * av_q2d(st->time_base), best_diff);

    return best;
}

/* -------------------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    // Binary mode on stdout
#ifdef _WIN32
    if (!isatty(fileno(stdout)))
        _setmode(_fileno(stdout), _O_BINARY);
#endif

    Args args;
    if (!parse_args(argc, argv, &args)) {
        usage(argv[0]);
        return 1;
    }

    av_log_set_level(AV_LOG_WARNING);

    // Initialize libplacebo
    pl_log log = pl_log_create(PL_API_VER, pl_log_params(
        .log_cb = pl_log_simple,
        .log_priv = stderr,
        .log_level = PL_LOG_WARN,  // Production: only warnings and errors
    ));

    pl_d3d11 d3d11 = pl_d3d11_create(log, pl_d3d11_params(
        .allow_software = true,
    ));

    if (!d3d11) {
        fprintf(stderr, "Failed to create D3D11 context\n");
        return 1;
    }
    pl_gpu gpu = d3d11->gpu;

    // Decode frame
    AVFrame *frame = decode_frame_at(args.input, args.pts);
    if (!frame) {
        fprintf(stderr, "Failed to decode frame at pts=%.3f\n", args.pts);
        return 1;
    }

    // Map AVFrame to GPU textures
    struct pl_frame pl_frame;
    pl_tex tex[4] = {0};  // Backing textures (will be created by pl_map_avframe)
    if (!pl_map_avframe(gpu, &pl_frame, tex, frame)) {
        fprintf(stderr, "Failed to map frame to GPU\n");
        av_frame_free(&frame);
        return 1;
    }

    // Extract features
    float features[PL_ML_FEATURE_DIM];
    struct pl_ml_feature_params params = {
        .target_nits = args.target_nits,
        .downsample_width = 256,
        .downsample_height = 144,
        .force_cpu_fallback = true,
        .debug_luma_path = args.debug_png,  // Save luma if requested
    };

    bool ok = pl_extract_ml_features(gpu, &pl_frame, &params, features);

    // Clean up mapped frame
    pl_unmap_avframe(gpu, &pl_frame);
    for (int i = 0; i < 4; i++)
        pl_tex_destroy(gpu, &tex[i]);
    av_frame_free(&frame);

    pl_d3d11_destroy(&d3d11);
    pl_log_destroy(&log);

    if (!ok) {
        fprintf(stderr, "Feature extraction failed\n");
        return 1;
    }

    // Write output
    FILE *out = stdout;
    if (args.output) {
        out = fopen(args.output, "wb");
        if (!out) {
            fprintf(stderr, "Cannot open output: %s\n", args.output);
            return 1;
        }
    }

    size_t written = fwrite(features, sizeof(float), PL_ML_FEATURE_DIM, out);

    if (args.output)
        fclose(out);

    if (written != PL_ML_FEATURE_DIM) {
        fprintf(stderr, "Failed to write features (wrote %zu/%d)\n",
                written, PL_ML_FEATURE_DIM);
        return 1;
    }

    fprintf(stderr, "Success: %d features written (%.2f KB)\n",
            PL_ML_FEATURE_DIM, (PL_ML_FEATURE_DIM * sizeof(float)) / 1024.0f);

    return 0;
}
