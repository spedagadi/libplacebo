/*
 * dv_render.c — headless Dolby Vision frame renderer via libplacebo + D3D11
 *
 * Decodes one frame from a DV video at a given PTS, renders it through the
 * full libplacebo colour pipeline under one of three tone-mapping modes, and
 * writes the result as raw RGB8 (24 bpp, row-major) to stdout.
 *
 * Modes:
 *   gold   — DV RPU polynomial (map_dowi=true, libplacebo applies RPU curve)
 *   spline — libplacebo pl_tone_map_spline driven by L1 metadata
 *   ml     — ML-predicted curve as a custom pl_tone_map_function via .cube LUT
 *
 * Usage:
 *   dv_render --input <file> --pts <seconds> --mode <gold|spline|ml>
 *             [--lut <file.cube>]   (required for --mode ml)
 *             [--width <px>] [--height <px>]
 *             [--out-nits <nits>]   (default 203, SDR white)
 *
 * Output: raw RGB8 to stdout, width*height*3 bytes.
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
#include <libavutil/pixdesc.h>

#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/d3d11.h>
#include <libplacebo/utils/libav.h>
#include <libplacebo/colorspace.h>
#include <libplacebo/tone_mapping.h>

#define PL_LIBAV_IMPLEMENTATION 1
#include <libplacebo/utils/libav.h>

/* -------------------------------------------------------------------------
 * Argument parsing
 * ---------------------------------------------------------------------- */
typedef struct {
    const char *input;
    double      pts;
    const char *mode;      /* "gold" | "spline" | "st2094-10" | "st2094-40" | "bt2390" | "ml" */
    const char *lut_file;  /* path to .cube, required for mode=ml */
    int         width;
    int         height;
    float       out_nits;
    float       l1_max_pq;       /* L1 DM block max PQ (0-1) */
    float       l1_avg_pq;       /* L1 DM block avg PQ (0-1) */
    /* Spline tone-mapper constants (PL_TONE_MAP_CONSTANTS defaults shown) */
    float       knee_adaptation; /* 0.0-1.0, default 0.4 */
    float       knee_minimum;    /* 0.0-0.5, default 0.1 */
    float       knee_maximum;    /* 0.5-1.0, default 0.8 */
    float       knee_default;    /* default 0.4 */
    float       slope_tuning;    /* 0-10,    default 1.5 */
    float       slope_offset;    /* 0-1,     default 0.2 */
    /* Gamut/colour volume controls */
    float       perceptual_strength; /* 0.0-1.0, default 0.8 — chroma restoration after tone map */
    int         gamut_expansion;     /* 0/1, default 0 — allow chroma expansion beyond source */
    float       spline_contrast; /* 0-1.5,   default 0.5 */
    /* Letterbox bar masking: zero output rows in bar region after rendering */
    float       top_bar_norm;   /* top_bar_pixels / source_height (0 = no bars) */
    float       bot_bar_norm;   /* bottom bar fraction */
} Args;

static void usage(const char *argv0)
{
    fprintf(stderr,
        "Usage: %s --input <file> --pts <sec> --mode <gold|spline|st2094-10|st2094-40|bt2390|ml>\n"
        "          [--lut <rpu_poly>]         required for --mode ml\n"
        "          [--width <px>] [--height <px>]\n"
        "          [--out-nits <nits>]         target display peak (default 203)\n"
        "          [--l1-max <0-1>]            frame peak PQ\n"
        "          [--l1-avg <0-1>]            frame avg PQ\n"
        "  Spline constants (leave unset for libplacebo defaults):\n"
        "          [--knee-adaptation <0-1>]   default 0.4\n"
        "          [--knee-minimum <0-0.5>]    default 0.1\n"
        "          [--knee-maximum <0.5-1>]    default 0.8\n"
        "          [--knee-default <val>]      default 0.4\n"
        "          [--slope-tuning <0-10>]     default 1.5\n"
        "          [--slope-offset <0-1>]      default 0.2\n"
        "          [--perceptual-strength <0-1>] default 0.8 (chroma restoration after tone map)\n"
        "          [--gamut-expansion <0|1>]    default 0 (allow chroma expansion beyond source)\n"
        "          [--spline-contrast <0-1.5>] default 0.5\n"
        "Output: raw RGB8 to stdout\n", argv0);
}

static bool parse_args(int argc, char **argv, Args *a)
{
    a->width    = 1920;
    a->height   = 1080;
    a->out_nits = 203.0f;
    /* Sentinel: -1 means "use libplacebo default" */
    a->knee_adaptation = -1.f;
    a->knee_minimum    = -1.f;
    a->knee_maximum    = -1.f;
    a->knee_default    = -1.f;
    a->slope_tuning    = -1.f;
    a->slope_offset         = -1.f;
    a->perceptual_strength  = -1.f;
    a->gamut_expansion      = -1;
    a->spline_contrast = -1.f;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--input")          && i+1 < argc) { a->input           = argv[++i]; }
        else if (!strcmp(argv[i], "--pts")            && i+1 < argc) { a->pts             = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--mode")           && i+1 < argc) { a->mode            = argv[++i]; }
        else if (!strcmp(argv[i], "--lut")            && i+1 < argc) { a->lut_file        = argv[++i]; }
        else if (!strcmp(argv[i], "--width")          && i+1 < argc) { a->width           = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--height")         && i+1 < argc) { a->height          = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--out-nits")       && i+1 < argc) { a->out_nits        = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--l1-max")         && i+1 < argc) { a->l1_max_pq       = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--l1-avg")         && i+1 < argc) { a->l1_avg_pq       = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--top-bar-norm")   && i+1 < argc) { a->top_bar_norm    = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--bot-bar-norm")   && i+1 < argc) { a->bot_bar_norm    = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--knee-adaptation") && i+1 < argc) { a->knee_adaptation = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--knee-minimum")   && i+1 < argc) { a->knee_minimum    = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--knee-maximum")   && i+1 < argc) { a->knee_maximum    = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--knee-default")   && i+1 < argc) { a->knee_default    = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--slope-tuning")   && i+1 < argc) { a->slope_tuning    = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--slope-offset")        && i+1 < argc) { a->slope_offset        = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--perceptual-strength") && i+1 < argc) { a->perceptual_strength = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--gamut-expansion")     && i+1 < argc) { a->gamut_expansion     = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--spline-contrast") && i+1 < argc) { a->spline_contrast = atof(argv[++i]); }
        else { fprintf(stderr, "Unknown argument: %s\n", argv[i]); return false; }
    }
    if (!a->input || !a->mode || a->pts < 0) {
        fprintf(stderr, "Missing required arguments.\n");
        return false;
    }
    if (!strcmp(a->mode, "ml") && !a->lut_file) {
        fprintf(stderr, "--mode ml requires --lut <file.cube>\n");
        return false;
    }
    return true;
}

/* -------------------------------------------------------------------------
 * ML RPU poly parser — kept below
 *
 * (old tone-map approach removed — ML now injects directly into comp[0])
 *
 * Reads a 1D .cube LUT (PQ in → PQ out, same scaling as pl_tone_map_spline).
 * The LUT is stored as an array of N (input, output) pairs in priv.
 * ---------------------------------------------------------------------- */

typedef struct {
    int    n;       /* number of LUT entries */
    float *xs;      /* input values  [0..n-1], normalised PQ */
    float *ys;      /* output values [0..n-1], normalised PQ */
} MlLut;

static void ml_tone_map(float *lut, const struct pl_tone_map_params *params)
{
    const MlLut *ml = params->function->priv;
    if (!ml || ml->n < 2) return;

    for (size_t i = 0; i < params->lut_size; i++) {
        float x = lut[i]; /* already scaled to params->input_max range */

        /* Normalise to [0,1] matching our ML LUT domain */
        float xn = (params->input_max > 0) ? (x / params->input_max) : x;
        xn = xn < 0 ? 0 : (xn > 1 ? 1 : xn);

        /* Linear interpolation into the ML LUT */
        float yn;
        if (xn <= ml->xs[0]) {
            yn = ml->ys[0];
        } else if (xn >= ml->xs[ml->n - 1]) {
            yn = ml->ys[ml->n - 1];
        } else {
            /* Binary search for the segment */
            int lo = 0, hi = ml->n - 1;
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (ml->xs[mid] <= xn) lo = mid; else hi = mid;
            }
            float t = (xn - ml->xs[lo]) / (ml->xs[hi] - ml->xs[lo]);
            yn = ml->ys[lo] + t * (ml->ys[hi] - ml->ys[lo]);
        }

        /* Scale output back to params->input_max range (same as spline) */
        lut[i] = yn * params->input_max;
    }
}

/* Parse a simple 2-column LUT file (x y per line, PQ normalised [0,1]).
 * Written by ml_inference.py when running on HDR10 content. */
static MlLut *load_ml_lut(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open ML LUT: %s\n", path); return NULL; }

    float *xs = malloc(4096 * sizeof(float));
    float *ys = malloc(4096 * sizeof(float));
    int n = 0;
    char line[256];
    while (n < 4096 && fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        float x, y;
        if (sscanf(line, "%f %f", &x, &y) == 2) {
            xs[n] = x;
            ys[n] = y;
            n++;
        }
    }
    fclose(f);
    if (n < 2) { free(xs); free(ys); return NULL; }

    MlLut *ml = malloc(sizeof(MlLut));
    ml->n  = n;
    ml->xs = xs;
    ml->ys = ys;
    return ml;
}

/* Parse RPU_POLY_1D file (written by dv_coef_model.py write_rpu_lut).
 * Returns false and leaves dovi->comp[0] untouched on failure. */
static bool parse_rpu_poly(const char *path, struct pl_dovi_metadata *dovi)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open RPU LUT: %s\n", path); return false; }

    char line[512];
    int  n_segs = 0;
    float pivots[9] = {0};
    int   n_pivots = 0;
    struct pl_reshape_data *comp = &dovi->comp[0];
    bool got_segs = false;

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "num_segs", 8) == 0) {
            sscanf(line, "num_segs %d", &n_segs);
            got_segs = true;
        } else if (strncmp(line, "pivots", 6) == 0) {
            char *p = line + 7;
            n_pivots = 0;
            while (n_pivots < 9 && sscanf(p, "%f", &pivots[n_pivots]) == 1) {
                n_pivots++;
                while (*p && *p != ' ' && *p != '\n') p++;
                while (*p == ' ') p++;
            }
        } else if (strncmp(line, "seg", 3) == 0) {
            int idx, order;
            float c0, c1, c2;
            if (sscanf(line, "seg %d %d %f %f %f", &idx, &order, &c0, &c1, &c2) == 5
                && idx < 8) {
                comp->method[idx]         = 0;  /* polynomial */
                comp->poly_coeffs[idx][0] = c0;
                comp->poly_coeffs[idx][1] = c1;
                comp->poly_coeffs[idx][2] = c2;
            }
        }
    }
    fclose(f);

    if (!got_segs || n_segs < 1 || n_pivots < 2) {
        fprintf(stderr, "Invalid RPU LUT file: %s\n", path);
        return false;
    }

    comp->num_pivots = (uint8_t)n_pivots;
    for (int i = 0; i < n_pivots && i < 9; i++)
        comp->pivots[i] = pivots[i];

    return true;
}

/* -------------------------------------------------------------------------
 * FFmpeg: decode one frame near the given PTS
 * ---------------------------------------------------------------------- */
static AVFrame *decode_frame_at(const char *path, double target_pts,
                                int out_w, int out_h)
{
    AVFormatContext *fmt = NULL;
    if (avformat_open_input(&fmt, path, NULL, NULL) < 0) {
        fprintf(stderr, "Cannot open: %s\n", path);
        return NULL;
    }
    avformat_find_stream_info(fmt, NULL);

    int vstream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (vstream < 0) { fprintf(stderr, "No video stream\n"); goto fail; }

    AVStream *st   = fmt->streams[vstream];
    const AVCodec *codec = avcodec_find_decoder(st->codecpar->codec_id);
    AVCodecContext *dec  = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(dec, st->codecpar);
    dec->export_side_data |= AV_CODEC_EXPORT_DATA_PRFT;
    if (avcodec_open2(dec, codec, NULL) < 0) {
        fprintf(stderr, "Cannot open decoder\n");
        avcodec_free_context(&dec);
        goto fail;
    }

    /* Seek to just before target PTS */
    int64_t seek_ts = (int64_t)(target_pts * AV_TIME_BASE);
    av_seek_frame(fmt, -1, seek_ts, AVSEEK_FLAG_BACKWARD);
    avcodec_flush_buffers(dec);

    AVPacket *pkt  = av_packet_alloc();
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
            double pts = frame->pts * av_q2d(st->time_base);
            double diff = pts - target_pts;
            /* Accept first frame at or past target, within 2 seconds */
            if (diff >= -0.05 && diff < best_diff) {
                best_diff = diff;
                av_frame_free(&best);
                best = av_frame_clone(frame);
                if (diff < 0.1) goto found; /* close enough */
            }
            av_frame_unref(frame);
        }
    }

found:
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&dec);
    avformat_close_input(&fmt);
    return best;

fail:
    avformat_close_input(&fmt);
    return NULL;
}

/* -------------------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    /* Binary mode on stdout — prevent Windows \n→\r\n translation of pixel data */
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    /* Ensure ffmpeg logs to stderr, not stdout */
    av_log_set_level(AV_LOG_WARNING);

    Args a = {0};
    if (!parse_args(argc, argv, &a)) { usage(argv[0]); return 1; }

    /* --- libplacebo log (stderr) --- */
    pl_log log = pl_log_create(PL_API_VER, pl_log_params(
        .log_cb    = pl_log_simple,
        .log_priv  = stderr,
        .log_level = PL_LOG_WARN,
    ));

    /* --- D3D11 GPU context (hardware, WARP software fallback) --- */
    pl_d3d11 d3d11 = pl_d3d11_create(log, pl_d3d11_params(
        .allow_software = true,
    ));
    if (!d3d11) { fprintf(stderr, "Failed to create D3D11 context\n"); return 1; }
    pl_gpu gpu = d3d11->gpu;

    /* --- Renderer --- */
    pl_renderer renderer = pl_renderer_create(log, gpu);

    /* --- Decode frame --- */
    AVFrame *avf = decode_frame_at(a.input, a.pts, a.width, a.height);
    if (!avf) { fprintf(stderr, "Failed to decode frame\n"); return 1; }

    /* --- Map AVFrame → pl_frame ---
     * Always use map_dovi=true so the DV RPU's custom ycc_to_rgb matrix
     * is applied. For spline/ml we then replace the reshaping curves with
     * identity — same matrix, different tone curve.
     */
    struct pl_frame image = {0};
    pl_tex tex[4] = {0};

    if (!pl_frame_recreate_from_avframe(gpu, &image, tex, avf)) {
        fprintf(stderr, "pl_frame_recreate_from_avframe failed\n");
        return 1;
    }

    bool ok = pl_map_avframe_ex(gpu, &image, pl_avframe_params(
        .frame    = avf,
        .tex      = tex,
        .map_dovi = true,   /* always — need the RPU colour matrix */
    ));
    if (!ok) { fprintf(stderr, "pl_map_avframe_ex failed\n"); return 1; }

    /* For spline/ml: keep the DV colour matrix but replace the RPU reshaping
     * curves with identity. Allocate a mutable copy of pl_dovi_metadata,
     * set each component to a pass-through polynomial, then point repr.dovi
     * at the copy. The original is const and must not be modified directly. */
    struct pl_dovi_metadata dovi_identity = {0};
    bool is_gold = !strcmp(a.mode, "gold");
    if (!is_gold &&
        image.repr.sys == PL_COLOR_SYSTEM_DOLBYVISION && image.repr.dovi)
    {
        /* Copy matrices and offsets from the original RPU */
        dovi_identity = *image.repr.dovi;

        /* Replace ONLY comp[0] (Y/luma) with identity — keep comp[1]/comp[2]
         * (Cb/Cr chroma) from the original RPU. The colorist's chroma grading
         * lives in comp[1] and comp[2]; stripping it causes the cool colour cast. */
        struct pl_reshape_data *y = &dovi_identity.comp[0];
        y->num_pivots        = 2;
        y->pivots[0]         = 0.0f;
        y->pivots[1]         = 1.0f;
        y->method[0]         = 0;     /* polynomial */
        y->poly_coeffs[0][0] = 0.0f;  /* c0: offset */
        y->poly_coeffs[0][1] = 1.0f;  /* c1: slope = 1 (pass-through) */
        y->poly_coeffs[0][2] = 0.0f;  /* c2: quadratic term */
        /* comp[1] and comp[2] are preserved from dovi_identity = *image.repr.dovi */
        image.repr.dovi = &dovi_identity;
    }

    /* --- Output texture (sRGB8) --- */
    pl_fmt out_fmt = pl_find_named_fmt(gpu, "rgba8");
    if (!out_fmt) out_fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 4, 8, 8,
                                        PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
    if (!out_fmt) { fprintf(stderr, "No suitable output format\n"); return 1; }

    pl_tex out_tex = pl_tex_create(gpu, pl_tex_params(
        .w            = a.width,
        .h            = a.height,
        .format       = out_fmt,
        .renderable   = true,
        .host_readable= true,
        .storable     = !!(out_fmt->caps & PL_FMT_CAP_STORABLE),
    ));
    if (!out_tex) { fprintf(stderr, "Failed creating output texture\n"); return 1; }

    /* --- Target frame --- */
    struct pl_frame target = {0};
    target.num_planes = 1;
    target.planes[0].texture = out_tex;
    target.planes[0].components = 3;
    target.planes[0].component_mapping[0] = 0; /* R */
    target.planes[0].component_mapping[1] = 1; /* G */
    target.planes[0].component_mapping[2] = 2; /* B */
    target.planes[0].component_mapping[3] = -1;
    target.crop = (pl_rect2df){ 0, 0, a.width, a.height };
    target.repr  = pl_color_repr_rgb;
    target.color = pl_color_space_srgb;

    /* Set target display peak luminance — drives tone mapper output range.
     * pl_color_space_srgb defaults to 203 nits (PL_COLOR_SDR_WHITE).
     * Overriding max_luma here tells libplacebo to target a different display:
     * e.g. 50 nits for a projector, 1000 nits for an HDR monitor.
     * The tone mapper (spline/ML) automatically adapts its knee and slope. */
    if (a.out_nits > 0 && a.out_nits != 203.0f)
        target.color.hdr.max_luma = a.out_nits;

    /* --- Tone mapping params --- */
    struct pl_color_map_params cmap = *(&pl_color_map_default_params);
    struct pl_render_params rparams = pl_render_default_params;
    rparams.color_map_params = &cmap;

    /* Apply user-supplied spline constants — negative sentinel means keep default */
#define APPLY_IF_SET(field, arg) if ((arg) >= 0.0f) cmap.tone_constants.field = (arg)
    APPLY_IF_SET(knee_adaptation, a.knee_adaptation);
    APPLY_IF_SET(knee_minimum,    a.knee_minimum);
    APPLY_IF_SET(knee_maximum,    a.knee_maximum);
    APPLY_IF_SET(knee_default,    a.knee_default);
    APPLY_IF_SET(slope_tuning,    a.slope_tuning);
    APPLY_IF_SET(slope_offset,    a.slope_offset);
    APPLY_IF_SET(spline_contrast, a.spline_contrast);
#undef APPLY_IF_SET

    /* Gamut/colour volume controls */
    if (a.perceptual_strength >= 0.0f)
        cmap.gamut_constants.perceptual_strength = a.perceptual_strength;
    if (a.gamut_expansion >= 0)
        cmap.gamut_expansion = (bool)a.gamut_expansion;


    if (is_gold) {
        /* map_dowi=true: libplacebo applies RPU polynomial directly.
         * Spline handles any residual HDR→SDR compression. */
        cmap.tone_mapping_function = &pl_tone_map_spline;
        cmap.metadata = PL_HDR_METADATA_CIE_Y;

    } else if (!strcmp(a.mode, "spline")    ||
               !strcmp(a.mode, "st2094-10") ||
               !strcmp(a.mode, "st2094-40") ||
               !strcmp(a.mode, "bt2390")) {
        /* Identity RPU reshape + chosen tone mapper driven by L1 values.
         *   spline    — single-pivot polynomial (libplacebo default)
         *   st2094-10 — SMPTE ST 2094-10 Annex B.2 rational EETF
         *   st2094-40 — SMPTE ST 2094-40 Annex B Bezier (uses HDR10+ ootf if present)
         *   bt2390    — ITU-R BT.2390 hermite spline EETF
         */
        if (!strcmp(a.mode, "st2094-10"))
            cmap.tone_mapping_function = &pl_tone_map_st2094_10;
        else if (!strcmp(a.mode, "st2094-40"))
            cmap.tone_mapping_function = &pl_tone_map_st2094_40;
        else if (!strcmp(a.mode, "bt2390"))
            cmap.tone_mapping_function = &pl_tone_map_bt2390;
        else
            cmap.tone_mapping_function = &pl_tone_map_spline;

        cmap.metadata = PL_HDR_METADATA_CIE_Y;
        if (a.l1_max_pq > 0) {
            image.color.hdr.max_pq_y = a.l1_max_pq;
            image.color.hdr.avg_pq_y = a.l1_avg_pq;
        }

    } else if (!strcmp(a.mode, "ml-lut")) {
        /* HDR10-compatible ML mode: apply ML polynomial as a 1D tone map LUT.
         * Works on any HDR10 source — no DV RPU context needed.
         * LUT format: one "x y" pair per line, PQ normalised [0,1].
         * Written by ml_write_lut() in ml_viewer.py. */
        if (!a.lut_file) {
            fprintf(stderr, "--mode ml-lut requires --lut <lut_file>\n");
            return 1;
        }
        MlLut *ml_lut = load_ml_lut(a.lut_file);
        if (!ml_lut) {
            fprintf(stderr, "Failed loading ML LUT: %s\n", a.lut_file);
            return 1;
        }
        static struct pl_tone_map_function ml_fn = {0};
        ml_fn.name        = "ml";
        ml_fn.description = "ML-predicted polynomial tone curve";
        ml_fn.scaling     = PL_HDR_PQ;
        ml_fn.map         = ml_tone_map;
        ml_fn.priv        = ml_lut;
        cmap.tone_mapping_function = &ml_fn;
        cmap.metadata = PL_HDR_METADATA_CIE_Y;
        if (a.l1_max_pq > 0) {
            image.color.hdr.max_pq_y = a.l1_max_pq;
            image.color.hdr.avg_pq_y = a.l1_avg_pq;
        }

    } else if (!strcmp(a.mode, "ml")) {
        /* ML mode: inject predicted RPU polynomial into comp[0] directly.
         * Runs at the SAME pipeline stage as DV gold — correct colour science.
         * comp[1]/comp[2] (chroma) are preserved from the original RPU. */
        if (!a.lut_file) {
            fprintf(stderr, "--mode ml requires --lut <rpu_poly file>\n");
            return 1;
        }
        if (!parse_rpu_poly(a.lut_file, &dovi_identity)) {
            fprintf(stderr, "Failed parsing ML RPU LUT\n");
            return 1;
        }
        /* Spline handles residual HDR→SDR after the RPU reshape */
        cmap.tone_mapping_function = &pl_tone_map_spline;
        cmap.metadata = PL_HDR_METADATA_CIE_Y;
        if (a.l1_max_pq > 0) {
            image.color.hdr.max_pq_y = a.l1_max_pq;
            image.color.hdr.avg_pq_y = a.l1_avg_pq;
        }
    }

    /* --- Render --- */
    ok = pl_render_image(renderer, &image, &target, &rparams);
    if (!ok) fprintf(stderr, "Warning: pl_render_image failed\n");

    /* --- Download result --- */
    size_t row_pitch = (size_t)a.width * 4;  /* RGBA8 = 4 bytes/pixel, no padding */
    size_t out_bytes = row_pitch * a.height;
    uint8_t *pixels = malloc(out_bytes);
    ok = pl_tex_download(gpu, pl_tex_transfer_params(
        .tex       = out_tex,
        .ptr       = pixels,
        .row_pitch = row_pitch,
    ));
    if (!ok) { fprintf(stderr, "pl_tex_download failed\n"); return 1; }

    /* Zero out letterbox bar rows — bars must stay pure black regardless of LUT.
     * top_bar_norm/bot_bar_norm are fractional (pixels / source_height).
     * We clamp to the rendered height so they work for any output resolution. */
    if (a.top_bar_norm > 0.0f || a.bot_bar_norm > 0.0f) {
        int top_rows = (int)(a.top_bar_norm * a.height + 0.5f);
        int bot_rows = (int)(a.bot_bar_norm * a.height + 0.5f);
        top_rows = top_rows < a.height ? top_rows : a.height;
        bot_rows = bot_rows < a.height ? bot_rows : a.height;
        for (int y = 0; y < top_rows; y++)
            memset(&pixels[y * a.width * 4], 0, (size_t)a.width * 4);
        for (int y = a.height - bot_rows; y < a.height; y++)
            memset(&pixels[y * a.width * 4], 0, (size_t)a.width * 4);
    }

    /* Write RGB8 to stdout (drop alpha channel) */
    for (int y = 0; y < a.height; y++) {
        for (int x = 0; x < a.width; x++) {
            fwrite(&pixels[(y * a.width + x) * 4], 1, 3, stdout);
        }
    }
    fflush(stdout);

    /* --- Cleanup --- */
    free(pixels);
    pl_unmap_avframe(gpu, &image);
    av_frame_free(&avf);
    pl_tex_destroy(gpu, &out_tex);
    pl_renderer_destroy(&renderer);
    pl_d3d11_destroy(&d3d11);
    pl_log_destroy(&log);
    return 0;
}
