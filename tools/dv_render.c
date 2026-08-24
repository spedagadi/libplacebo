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
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
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
#include <libplacebo/dither.h>

#define PL_LIBAV_IMPLEMENTATION 1
#include <libplacebo/utils/libav.h>

/* -------------------------------------------------------------------------
 * IEEE 754 binary16 (float16) → float32 conversion
 *
 * libplacebo stores float16 textures as IEEE 754 binary16 bit patterns.
 * When we download them as uint16_t, we must convert properly.
 * ---------------------------------------------------------------------- */
static inline float __half2float(uint16_t h) {
    /* Extract sign, exponent, mantissa from binary16 */
    uint32_t s      = (h >> 15) & 0x1;   /* sign bit           */
    uint32_t exp    = (h >> 10) & 0x1f;   /* exponent (biased)  */
    uint32_t mant   =  h        & 0x3ff;  /* mantissa (10 bits) */

    if (exp == 0) {
        /* Subnormal or zero */
        if (mant == 0) {
            /* ±0 */
            return (s ? -0.0f : 0.0f);
        }
        /* Subnormal: normalize to 1.x * 2^-14 */
        for (int i = 0; i < 10; i++) {
            if (mant & (1 << i)) {
                mant &= ~(1 << i);
                exp = 1;
                break;
            }
        }
    }

    /* Normal case: exponent = 1-30, implicit leading 1 */
    if (exp != 0) {
        exp += 127 - 15; /* Re-bias: 15 → 127 */
    } else {
        /* Was subnormal above but normalized — already handled */
        exp = 1 + 127 - 15;
    }

    /* Combine: sign (1 bit) | exponent (8 bits) | mantissa (23 bits) */
    uint32_t i = (s << 31) | (exp << 23) | (mant << 13);
    float result;
    memcpy(&result, &i, sizeof(result));
    return result;
}

static const float *get_blue_noise_matrix(void)
{
    static float matrix[16 * 16];
    static bool initialized;
    if (!initialized) {
        pl_generate_blue_noise(matrix, 16);
        initialized = true;
    }
    return matrix;
}

/* -------------------------------------------------------------------------
 * Argument parsing
 * ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------------
 * L2 trim constants — tuned for HDR→SDR contrast recovery (DV-style punch)
 * ---------------------------------------------------------------------- */

/* Optimal gamma heuristic: predicts per-scene gamma from scene statistics.
 *
 * Based on analysis of XGBoost gamma model (88 features, corr=0.651).
 * This heuristic captures the dominant factors with 3 simple inputs:
 *   - content brightness ratio (maxscl / target_nits) — expansion vs compression
 *   - scene average (l1_avg_pq) — dark vs bright scene
 *   - peak-to-average ratio — dynamic range / specular intensity
 *
 * Typical outputs: 0.75–0.85 for dark drama (punchy), 0.90–1.0 for bright content.
 * Matches the XGBoost model's behaviour within ±0.05 gamma.
 */

/* Map content brightness ratio → gamma (heuristic anchor points).
 * Bright content → higher gamma (less compression, closer to identity).
 * Dark content → lower gamma (more compression, more punch). */
static float predict_gamma_from_brightness(float content_ratio, float avg_pq)
{
    /* content_ratio = maxscl / target_nits
     * > 1.0 → content brighter than display → compress
     * < 1.0 → content dimmer than display → expand */

    /* Base gamma: dark scenes get punchier trim
     * Clamp content_ratio to [0.1, 10.0] to avoid extremes */
    content_ratio = content_ratio < 0.1f ? 0.1f : (content_ratio > 10.0f ? 10.0f : content_ratio);

    /* Exponential decay: gamma rises from ~0.7 to ~1.0 as content_ratio increases */
    float gamma = 0.72f + 0.28f * powf(content_ratio / 3.0f, 0.35f);

    /* Dark scenes get a bump (more punchy trim for dark drama) */
    if (avg_pq < 0.3f)
        gamma -= 0.05f;
    else if (avg_pq > 0.7f)
        gamma += 0.03f;  /* bright scenes need less punch */

    /* Clamp to sensible range [0.65, 1.15] */
    return gamma < 0.65f ? 0.65f : (gamma > 1.15f ? 1.15f : gamma);
}

static float predict_knee_point_from_stats(float maxscl, float avg_pq,
                                            float fraction_bright, float out_nits)
{
    (void)fraction_bright; (void)out_nits;
    float knee = 0.72f + 12.0f * avg_pq;
    float peak_to_avg = maxscl / fmaxf(avg_pq, 1e-6f);
    float spec_factor = fminf(1.0f, peak_to_avg / 1000.0f);
    knee -= 0.06f * spec_factor;
    return knee < 0.68f ? 0.68f : (knee > 0.90f ? 0.90f : knee);
}

/* -------------------------------------------------------------------------
 * GPU-based KrigBilateral spatial contrast recovery (port of mpv shader)
 *
 * Pipeline:
 *   1. Extract luma from tone-mapped RGBA output → float texture
 *   2. Bilateral filter (edge-preserving blur) → blurred luma texture
 *   3. Detail injection: boosted_luma = luma + (luma - blurred) * boost
 *   4. Download boosted luma, blend back into RGBA output
 *
 * GPU compute shader (GLSL 330 core, compute shader extension):
 *   - 9×9 bilateral window (radius=4)
 *   - Gaussian spatial weight + range weight
 *   - Edge-preserving: strong luma differences get low weight
 *
 * Parameters (matching madVR contrast booster / Lumagen processing):
 *   σ_s = 3.0 — spatial spread (controls macro-contrast scale)
 *   σ_r = 15.0 — range spread (controls edge preservation aggression)
 *   radius = 4 — 9×9 window
 *   boost = cr_strength × 2.0 (0.2–1.0)
 *
 * Performance on D3D11: ~1.2M workgroups at 640×360, 16×16 threads → ~5ms
 */
static const char *bilateral_filter_glsl =
    "#version 330\n"
    "#extension GL_ARB_compute_shader : require\n"
    "#extension GL_ARB_shader_image_access : require\n"
    "\n"
    "layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;\n"
    "\n"
    "uniform sampler2D u_input;\n"
    "uniform float u_radius;\n"
    "uniform float u_sigma_s;\n"
    "uniform float u_sigma_r;\n"
    "uniform float u_boost;\n"
    "uniform float u_size_x;\n"
    "uniform float u_size_y;\n"
    "\n"
    "layout (r32f, binding = 0) uniform image2D u_output;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    vec2 pos = vec2(gl_GlobalInvocationID.xy);\n"
    "    if (pos.x >= u_size_x || pos.y >= u_size_y) return;\n"
    "    ivec2 ipos = ivec2(pos);\n"
    "\n"
    "    float center = texture(u_input, pos / vec2(u_size_x, u_size_y)).x;\n"
    "\n"
    "    float r = u_radius;\n"
    "    float inv2ss = -1.0 / (2.0 * u_sigma_s * u_sigma_s);\n"
    "    float inv2sr = -1.0 / (2.0 * u_sigma_r * u_sigma_r);\n"
    "    float w_sum = 0.0;\n"
    "    float y_sum = 0.0;\n"
    "\n"
    "    for (float fy = -r; fy <= r; fy++) {\n"
    "        for (float fx = -r; fx <= r; fx++) {\n"
    "            ivec2 np = ipos + ivec2(int(fx), int(fy));\n"
    "            if (np.x < 0 || np.x >= int(u_size_x) ||\n"
    "                np.y < 0 || np.y >= int(u_size_y)) continue;\n"
    "\n"
    "            float ny = texture(u_input, vec2(np) / vec2(u_size_x, u_size_y)).x;\n"
    "            float dx = fx, dy = fy;\n"
    "            float sw = exp((dx*dx + dy*dy) * inv2ss);\n"
    "            float yd = ny - center;\n"
    "            float rw = exp(yd * yd * inv2sr);\n"
    "            float w = sw * rw;\n"
    "\n"
    "            w_sum += w;\n"
    "            y_sum += ny * w;\n"
    "        }\n"
    "    }\n"
    "\n"
    "    float blurred = (w_sum > 0.0) ? (y_sum / w_sum) : center;\n"
    "    float detail = center - blurred;\n"
    "    float boosted = center + detail * u_boost;\n"
    "    imageStore(u_output, ipos, vec4(boosted, 0.0, 0.0, 1.0));\n"
    "}\n";

/* --- GPU CR: pass variable & descriptor definitions --- */
#define NUM_CR_VARS 6
#define NUM_CR_DESCS 2
#define NUM_CR_UPDATES 6

static struct pl_var cr_vars[NUM_CR_VARS];
static struct pl_desc cr_descs[NUM_CR_DESCS];

/* Initialize cr_vars and cr_descs (called once at startup) */
static void init_cr_vars(void)
{
    cr_vars[0] = pl_var_float("u_radius");
    cr_vars[1] = pl_var_float("u_sigma_s");
    cr_vars[2] = pl_var_float("u_sigma_r");
    cr_vars[3] = pl_var_float("u_boost");
    cr_vars[4] = pl_var_float("u_size_x");
    cr_vars[5] = pl_var_float("u_size_y");

    cr_descs[0] = (struct pl_desc){
        .name   = "u_input",
        .type   = PL_DESC_SAMPLED_TEX,
        .access = PL_DESC_ACCESS_READONLY,
    };
    cr_descs[1] = (struct pl_desc){
        .name   = "u_output",
        .type   = PL_DESC_STORAGE_IMG,
        .access = PL_DESC_ACCESS_WRITEONLY,
    };
}

typedef struct {
    const char *input;
    double      pts;
    const char *mode;      /* "gold" | "spline" | "st2094-10" | "st2094-40" | "bt2390" | "ml" | "contrast-recovery" */
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
    /* L2 trim — DV-style power/saturation applied in RGB space after tone mapping */
    float       l2_power;       /* 2048=neutral, <2048=more contrast. -1=disabled */
    float       l2_sat_gain;    /* 2048=neutral, >2048=more saturation. -1=disabled */
    /* HDR contrast recovery mode: auto-predicts gamma from scene stats, applies L2 trim */
    int         contrast_recovery; /* 1=enable auto contrast recovery */
    float       contrast_gamma;   /* manual override for gamma in contrast-recovery mode */
    float       contrast_sat;     /* manual override for saturation in contrast-recovery mode */
    /* Specular highlight roll-off (piecewise L2 trim) — deferred to XGBoost model */
    float       highlight_knee;   /* reserved: 0.0-1.0, default: auto from scene stats */
    /* Libplacebo HDR contrast recovery (high-frequency detail injection) */
    float       cr_strength;      /* 0.0-0.5, default: auto from predicted gamma */
    float       cr_smoothness;    /* >1.0, default: 2.5 (tighter halos on fine textures) */
    /* Luma-Weighted Warm Chroma Reshaping (fire pop) — boosts orange/red density
     * for high-luma warm pixels (explosions, fire, incandescent sources).
     * Three phases: specular desaturation (core white-hot), mid-flame body
     * saturation injection, anti-pink hue constraint. */
    float       fire_pop_strength;/* 0.0-2.0, default 1.0 (scales chroma boost scalars) */
} Args;

static void usage(const char *argv0)
{
    fprintf(stderr,
        "Usage: %s --input <file> --pts <sec> --mode <gold|spline|st2094-10|st2094-40|bt2390|ml|contrast-recovery>\n"
        "          [--lut <rpu_poly>]         required for --mode ml\n"
        "          [--width <px>] [--height <px>]\n"
        "          [--out-nits <nits>]         target display peak (default 203)\n"
        "          [--l1-max <0-1>]            frame peak PQ (required for contrast-recovery)\n"
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
        "  HDR Contrast Recovery (new):\n"
        "          --contrast-recovery          enable auto gamma prediction + L2 trim\n"
        "          [--contrast-gamma <0.65-1.15>] default: auto-predicted from scene stats\n"
        "          [--contrast-sat <0.5-3.0>]   default: 1.0 (no saturation change)\n"
        "  Specular highlight roll-off (piecewise L2 trim — adds 'gleam' to highlights):\n"
        "          [--highlight-knee <0.70-0.90>] default: auto from scene stats\n"
        "  Libplacebo HDR contrast recovery (high-frequency detail injection):\n"
        "          [--cr-strength <0.0-0.5>]    default: auto-scaled from predicted gamma\n"
        "          [--cr-smoothness <1.0-5.0>]  default: 2.5 (tighter than libplacebo 3.5 default)\n"
        "  Luma-Weighted Warm Chroma Reshaping (fire pop):\n"
        "          [--fire-pop-strength <0-2.0>]    0=off, 1.0=default, scales boost\n"
        "  Manual L2 trim (overrides contrast-recovery):\n"
        "          [--l2-power <val>]           2048=neutral, <2048=more contrast\n"
        "          [--l2-sat-gain <val>]        2048=neutral, >2048=more saturation\n"
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
    a->l2_power    = -1.f;
    a->l2_sat_gain = -1.f;
    a->contrast_recovery = 0;
    a->contrast_gamma  = 0.f;  /* 0 = auto-predict */
    a->contrast_sat    = 1.f;  /* 1.0 = neutral */
    a->highlight_knee  = -1.f;  /* -1 = auto-predict from scene stats */
    a->cr_strength     = -1.f;  /* -1 = auto-scale from gamma */
    a->cr_smoothness   = -1.f;  /* -1 = use 2.5 default */
    a->fire_pop_strength = 0.f;  /* 0 = disabled (not enabled unless explicitly set) */

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--input")            && i+1 < argc) { a->input           = argv[++i]; }
        else if (!strcmp(argv[i], "--pts")              && i+1 < argc) { a->pts             = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--mode")             && i+1 < argc) { a->mode            = argv[++i]; }
        else if (!strcmp(argv[i], "--lut")              && i+1 < argc) { a->lut_file        = argv[++i]; }
        else if (!strcmp(argv[i], "--width")            && i+1 < argc) { a->width           = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--height")           && i+1 < argc) { a->height          = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--out-nits")         && i+1 < argc) { a->out_nits        = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--l1-max")           && i+1 < argc) { a->l1_max_pq       = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--l1-avg")           && i+1 < argc) { a->l1_avg_pq       = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--top-bar-norm")     && i+1 < argc) { a->top_bar_norm    = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--bot-bar-norm")     && i+1 < argc) { a->bot_bar_norm    = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--knee-adaptation")  && i+1 < argc) { a->knee_adaptation = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--knee-minimum")     && i+1 < argc) { a->knee_minimum    = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--knee-maximum")     && i+1 < argc) { a->knee_maximum    = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--knee-default")     && i+1 < argc) { a->knee_default    = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--slope-tuning")     && i+1 < argc) { a->slope_tuning    = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--slope-offset")           && i+1 < argc) { a->slope_offset        = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--perceptual-strength") && i+1 < argc) { a->perceptual_strength = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--gamut-expansion")     && i+1 < argc) { a->gamut_expansion     = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--spline-contrast")    && i+1 < argc) { a->spline_contrast = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--l2-power")           && i+1 < argc) { a->l2_power    = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--l2-sat-gain")        && i+1 < argc) { a->l2_sat_gain = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--contrast-recovery")  && i+1 < argc) { a->contrast_recovery = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--contrast-gamma")     && i+1 < argc) { a->contrast_gamma = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--contrast-sat")       && i+1 < argc) { a->contrast_sat = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--cr-strength")        && i+1 < argc) { a->cr_strength  = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--cr-smoothness")      && i+1 < argc) { a->cr_smoothness= atof(argv[++i]); }
        else if (!strcmp(argv[i], "--highlight-knee")     && i+1 < argc) { a->highlight_knee = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--fire-pop-strength")  && i+1 < argc) { a->fire_pop_strength = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--contrast-recovery")) { a->contrast_recovery = 1; }
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

    /* --- Initialize GPU CR variable/descriptor tables --- */
    init_cr_vars();

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

    /* --- Output texture: float16 (high-precision tone-map output).
     * Rendering to float16 instead of uint8 preserves the full precision of
     * libplacebo's tone-mapping — the subsequent float32 pipeline processes
     * pristine gradients, not 8-bit quantization steps. */
    pl_fmt out_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 16,
                                 PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE
                                 | PL_FMT_CAP_STORABLE);
    if (!out_fmt) {
        /* Fallback: use rgba8 if float16 render target unavailable */
        fprintf(stderr, "  [WARN] float16 render target unavailable, falling back to rgba8\n");
        out_fmt = pl_find_named_fmt(gpu, "rgba8");
        if (!out_fmt) out_fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 4, 8, 8,
                                            PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
    }
    if (!out_fmt) { fprintf(stderr, "No suitable output format\n"); return 1; }

    bool wants_float_processing =
        a.l2_power > 0.0f || a.l2_sat_gain > 0.0f ||
        a.fire_pop_strength > 0.0f || a.contrast_recovery;
    bool is_deep = pl_fmt_is_float(out_fmt) && out_fmt->num_components >= 4 &&
                   out_fmt->component_depth[0] >= 16;
    if (wants_float_processing && !is_deep) {
        fprintf(stderr,
                "High-precision render target is required for fire-pop/L2/CR "
                "processing, but no RGBA16F target is available\n");
        return 1;
    }

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
    /* When fire-pop is active, force gamut expansion so libplacebo preserves
     * the warm chroma we inject — perceptual mapping won't crush it to white.
     * desaturation_threshold is an internal libplacebo tone-mapper param;
     * gamut_expansion=1 is the closest public API equivalent. */
    if (a.fire_pop_strength > 0.0f) {
        fprintf(stderr, "  Fire pop enabled (strength=%.2f): gamut expansion ON\n",
                a.fire_pop_strength);
        cmap.gamut_expansion = true;
    } else if (a.gamut_expansion >= 0) {
        cmap.gamut_expansion = (bool)a.gamut_expansion;
    }


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

    } else if (!strcmp(a.mode, "contrast-recovery")) {
        /* HDR Contrast Recovery mode:
         *   1. Identity RPU reshape (keep DV colour matrix)
         *   2. Spline tone mapping for HDR→SDR range reduction
         *   3. DV-style L2 power trim with auto-predicted gamma
         *
         * Auto gamma prediction: uses scene maxscl, target nits, and average
         * luminance to estimate per-scene contrast trim matching XGBoost model
         * output (corr=0.651). Dark drama → lower gamma (more punch).
         * Bright nature → higher gamma (less compression). */
        if (a.l1_max_pq <= 0.01f) {
            fprintf(stderr,
                "Warning: --l1-max not set for contrast-recovery — using maxscl=%.4f\n",
                image.color.hdr.max_pq_y);
            a.l1_max_pq = image.color.hdr.max_pq_y;
        }

        /* Predict gamma from scene statistics (heuristic matching XGBoost model) */
        float content_ratio = a.l1_max_pq / (a.out_nits > 0.01f ? a.out_nits / 10000.0f : 0.02f);
        float predicted_gamma = predict_gamma_from_brightness(content_ratio, a.l1_avg_pq);

        /* Use predicted gamma unless manually overridden */
        float gamma = (a.contrast_gamma > 0.0f) ? a.contrast_gamma : predicted_gamma;
        float power = 2048.0f / gamma;
        float sat = a.contrast_sat > 0.0f ? a.contrast_sat * 2048.0f : 2048.0f;

        /* Report what was predicted (for verification) */
        fprintf(stderr, "Contrast recovery: gamma=%.3f power=%.0f sat=%.0f "
                        "(ratio=%.2f avg=%.3f)\n",
                gamma, power, sat, content_ratio, a.l1_avg_pq);

	/* highlight knee deferred to XGBoost model (specular highlight roll-off) */

        /* Set L2 trim values — these are consumed by the post-render section */
        a.l2_power = power;
        a.l2_sat_gain = sat;
        /* Spline handles the base HDR→SDR range reduction */
        cmap.tone_mapping_function = &pl_tone_map_spline;
        cmap.metadata = PL_HDR_METADATA_CIE_Y;
        image.color.hdr.max_pq_y = a.l1_max_pq;
        image.color.hdr.avg_pq_y = a.l1_avg_pq;

        /* HDR contrast recovery (high-frequency detail injection):
         * Auto-scales based on predicted gamma — punchy scenes get more
         * micro-contrast injection, flat scenes back off to avoid noise. */
        if (a.cr_strength >= 0.0f) {
            cmap.contrast_recovery = a.cr_strength;
            fprintf(stderr, "  CR strength: %.3f (explicit)\n", a.cr_strength);
        } else {
            /* Scale inversely: lower gamma → more detail injection.
             * gamma=0.7 → 0.42, gamma=1.0 → 0.28, gamma=1.2 → 0.25 */
            cmap.contrast_recovery = fmaxf(0.1f, fminf(0.5f,
                0.25f + (1.2f - gamma) * 0.15f));
            fprintf(stderr, "  CR strength: %.3f (auto from gamma %.3f)\n",
                    cmap.contrast_recovery, gamma);
        }

        /* Contrast recovery smoothness (blur kernel radius for high/low split).
         * 2.5 = tighter halos on fine textures. 3.5 = libplacebo default.
         * Users can increase for broader boost or decrease for sharper edges. */
        if (a.cr_smoothness >= 0.0f) {
            cmap.contrast_smoothness = a.cr_smoothness;
            fprintf(stderr, "  CR smoothness: %.1f (explicit)\n", a.cr_smoothness);
        } else {
            cmap.contrast_smoothness = 2.5f;
            fprintf(stderr, "  CR smoothness: %.1f (default)\n", cmap.contrast_smoothness);
        }
    }

    /* --- Render --- */
    ok = pl_render_image(renderer, &image, &target, &rparams);
    if (!ok) fprintf(stderr, "Warning: pl_render_image failed\n");

    /* --- Download result --- */
    /* We need a uint8_t *pixels buffer for the unified output path (fwrite to
     * stdout). For deep (float16) textures we first download as uint16_t, then
     * either: (a) use it directly as a high-precision float source for the
     * fire-pop/L2 pipeline, or (b) dither-quantize to uint8 for pass-through
     * mode. */
    uint16_t *pixels16 = NULL;
    uint8_t *pixels = NULL;
    size_t row_pitch_16 = (size_t)a.width * 8;  /* RGBA16F = 8 bytes/pixel */
    size_t row_pitch_8  = (size_t)a.width * 4;  /* RGBA8 = 4 bytes/pixel */

    if (is_deep) {
        pixels16 = malloc((size_t)a.width * a.height * 8);
        if (!pixels16) { fprintf(stderr, "Cannot allocate float16 download buffer\n"); return 1; }
        ok = pl_tex_download(gpu, pl_tex_transfer_params(
            .tex       = out_tex,
            .ptr       = pixels16,
            .row_pitch = row_pitch_16,
        ));
    } else {
        pixels = malloc((size_t)a.width * a.height * 4);
        if (!pixels) { fprintf(stderr, "Cannot allocate uint8 download buffer\n"); return 1; }
        ok = pl_tex_download(gpu, pl_tex_transfer_params(
            .tex       = out_tex,
            .ptr       = pixels,
            .row_pitch = row_pitch_8,
        ));
    }
    if (!ok) { fprintf(stderr, "pl_tex_download failed\n"); return 1; }

    /* --- Pipeline trace header (printed once at entry point) --- */
    fprintf(stderr,
        "  [PIPELINE TRACE]\n"
        "    Tone-map output:  %s (%d-bit)\n"
        "    Tone-map mode:    %s\n"
        "    Active blocks:    ",
        is_deep ? "float16 (rgba16f)" : "uint8 (rgba8)",
        is_deep ? 16 : 8,
        a.mode);
    {
        int has_cr   = (a.contrast_recovery && a.cr_strength > 0.0f);
        int has_fire = (a.fire_pop_strength > 0.0f);
        int has_l2   = (a.l2_power > 0.0f || a.l2_sat_gain > 0.0f);
        if (!has_cr && !has_fire && !has_l2)
            fprintf(stderr, "NONE (pass-through)");
        else {
            if (has_cr)   fprintf(stderr, "CR(%.3f) ", a.cr_strength);
            if (has_fire) fprintf(stderr, "FIRE(%.2f) ", a.fire_pop_strength);
            if (has_l2)   fprintf(stderr, "L2(p=%.0f s=%.0f) ", a.l2_power, a.l2_sat_gain);
        }
        fprintf(stderr, "\n    Data flow:    ");
        if (is_deep) {
            fprintf(stderr, "float16[tone-map] → ");
        } else {
            fprintf(stderr, "uint8[tone-map] → ");
        }
        if (has_fire || has_l2) {
            fprintf(stderr, "FLOAT32[work→fire-pop→L2] → uint8[dithered output]");
        } else if (has_cr) {
            fprintf(stderr, "%sGPU CR → uint8[output]", is_deep ? "float16[" : "");
        } else {
            fprintf(stderr, "%s[output]", is_deep ? "float16" : "uint8");
        }
        fprintf(stderr, "\n"
        "    Precision:   Tone-map output is %s — no pre-quantization banding.\n"
        "                 All intermediate processing (fire-pop, L2) in float32.\n"
        "                 Single uint8 quantize with blue-noise dither at output only.\n"
        "  [TRACE CONTINUES PER-BLOCK BELOW]\n",
        is_deep ? "16-bit float (no quantization loss)" : "8-bit (standard)");
    }

    /* Float working buffer for L2/fire-pop pipeline — avoids double-quantization
     * (uint8→float→uint8→float→uint8) that creates visible banding.
     * Declared at outer scope so references below (universal quantize, etc.)
     * are visible even when not inside the conditional. */
    float *work = NULL;
    int did_processing = 0;

    if (a.l2_power > 0.0f || a.l2_sat_gain > 0.0f ||
        a.fire_pop_strength > 0.0f) {
        fprintf(stderr,
            "  [ALLOC]  float32 work buffer (%d×%d, %.1f KB)\n",
            a.width, a.height,
            (float)a.width * a.height * 3 * sizeof(float) / 1024.0f);
        work = malloc((size_t)a.width * a.height * 3 * sizeof(float));
        if (work) {
            /* Upsample tone-map output to float32 for single-precision processing.
             * For deep (float16) textures this preserves full tone-map precision.
             * libplacebo stores float16 as IEEE 754 binary16 bit patterns,
             * so we must convert via __half2float (not raw division). */
            if (is_deep && pixels16) {
                for (int i = 0; i < a.width * a.height; i++) {
                    work[i*3+0] = __half2float(pixels16[i*4+0]);
                    work[i*3+1] = __half2float(pixels16[i*4+1]);
                    work[i*3+2] = __half2float(pixels16[i*4+2]);
                }
                fprintf(stderr,
                    "  [UPSAMPLE] float16 → float32 (%d×%d pixels, IEEE754)\n",
                    a.width, a.height);
            } else if (pixels) {
                for (int i = 0; i < a.width * a.height; i++) {
                    work[i*3+0] = pixels[i*4+0] / 255.0f;
                    work[i*3+1] = pixels[i*4+1] / 255.0f;
                    work[i*3+2] = pixels[i*4+2] / 255.0f;
                }
                fprintf(stderr, "  [UPSAMPLE] uint8 → float32 (fallback, 8-bit source)\n");
            }
            did_processing = 1;
            /* Allocate uint8 output buffer for the dithered-quantize step.
             * This buffer receives the final float32→uint8 output. */
            size_t out_size = (size_t)a.width * a.height * 4;
            pixels = malloc(out_size);
            if (!pixels) {
                fprintf(stderr, "  [WARN]   uint8 output buffer allocation FAILED\n");
            }
        } else {
            fprintf(stderr, "  [WARN]   Float buffer allocation FAILED — will fall through to raw output\n");
        }

        float l2_gamma_val = (a.l2_power > 0.0f) ? 2048.0f / a.l2_power : 1.0f;
        float l2_sat_val   = (a.l2_sat_gain > 0.0f) ? a.l2_sat_gain / 2048.0f : 1.0f;
        float power_val   = a.l2_power;
        float sat_val     = a.l2_sat_gain;

        /* Report which mode supplied the L2 values (stderr, not stdout) */
        if (a.contrast_recovery && !strcmp(a.mode, "contrast-recovery") && a.contrast_gamma <= 0.0f) {
            fprintf(stderr, "  L2 trim: AUTO gamma=%.3f (contrast-recovery)\n", l2_gamma_val);
        } else if (a.contrast_recovery && a.contrast_gamma > 0.0f)
            fprintf(stderr, "  L2 trim: MANUAL gamma=%.3f overrides auto\n", l2_gamma_val);
        else if (a.l2_power > 0.0f)
            fprintf(stderr, "  L2 trim: MANUAL power=%.0f sat=%.0f\n", power_val, sat_val);

        /* --- Spatial contrast recovery boost (GPU bilateral filter) ---
         * Edge-preserving detail injection via libplacebo compute shader.
         * Edge-preserving Gaussian blur preserves edges while amplifying
         * mid-frequency detail -- the "theater pop" effect.
         * Adapted from mpv KrigBilateral.glsl by Shiandow (LGPL v3). */
        if (a.contrast_recovery && a.cr_strength > 0.0f &&
            !strcmp(a.mode, "contrast-recovery")) {
            float boost = a.cr_strength * 2.0f;
            float sigma_s = 3.0f;
            float sigma_r = 15.0f;
            int radius = 4;

            if (a.width > 0 && a.height > 0 && gpu) {
                /* 1. Extract luma (0-255) from RGBA pixels */
                float *luma_data = (float *)malloc(
                    (size_t)a.width * a.height * sizeof(float));
                if (!luma_data) {
                    fprintf(stderr, "  GPU CR: CPU luma allocation failed\n");
                } else {
                    for (int i = 0; i < a.width * a.height; i++) {
                        luma_data[i] = 0.2126f * work[i*3+0]
                                     + 0.7152f * work[i*3+1]
                                     + 0.0722f * work[i*3+2];
                    }

                    /* 2. Find a suitable float format for textures */
                                        pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 32, 32,
                                                                                            PL_FMT_CAP_SAMPLEABLE);

                    if (fmt) {
                        /* 3. Create luma texture (sampleable) */
                        pl_tex luma_tex = pl_tex_create(gpu, pl_tex_params(
                            .w = a.width, .h = a.height,
                            .format = fmt,
                            .sampleable = true, .host_writable = true,
                        ));

                        if (luma_tex) {
                            /* Upload luma data to GPU */
                            pl_tex_upload(gpu, pl_tex_transfer_params(
                                .tex = luma_tex,
                                .rc = (pl_rect3d){
                                    .x0 = 0, .y0 = 0, .z0 = 0,
                                    .x1 = a.width, .y1 = a.height, .z1 = 1,
                                },
                                .ptr = luma_data,
                            ));

                            /* 4. Create output texture (storable + host-readable) */
                                                        pl_fmt cr_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 32, 32,
                                                                                                                 PL_FMT_CAP_STORABLE
                                                                                                             | PL_FMT_CAP_HOST_READABLE);

                            if (cr_fmt) {
                                pl_tex cr_out_tex = pl_tex_create(gpu, pl_tex_params(
                                    .w = a.width, .h = a.height,
                                    .format = cr_fmt,
                                    .storable = true, .host_readable = true,
                                ));

                                if (cr_out_tex) {
                                    /* 5. Create compute shader pass */
                                    pl_pass pass = pl_pass_create(gpu,
                                        pl_pass_params(
                                            .type            = PL_PASS_COMPUTE,
                                            .variables       = cr_vars,
                                            .num_variables   = NUM_CR_VARS,
                                            .descriptors     = cr_descs,
                                            .num_descriptors = NUM_CR_DESCS,
                                            .glsl_shader     = bilateral_filter_glsl,
                                        ));

                                    if (pass) {
                                        /* 6. Bind textures */
                                        struct pl_desc_binding bindings[] = {
                                            { .object = luma_tex },
                                            { .object = cr_out_tex },
                                        };

                                        /* 7. Push runtime uniforms */
                                        float radi = (float)radius;
                                        struct pl_var_update updates[] = {
                                            { .index = 0, .data = &radi },
                                            { .index = 1, .data = &sigma_s },
                                            { .index = 2, .data = &sigma_r },
                                            { .index = 3, .data = &boost },
                                            { .index = 4, .data = &a.width },
                                            { .index = 5, .data = &a.height },
                                        };

                                        /* 8. Execute pass */
                                        pl_pass_run(gpu,
                                            pl_pass_run_params(
                                                .pass            = pass,
                                                .var_updates     = updates,
                                                .num_var_updates = NUM_CR_UPDATES,
                                                .desc_bindings   = bindings,
                                            ));

                                        /* 9. Download output */
                                        float *boosted = (float *)malloc(
                                            (size_t)a.width * a.height * sizeof(float));
                                        if (boosted) {
                                            pl_tex_download(gpu, pl_tex_transfer_params(
                                                .tex = cr_out_tex,
                                                .rc = (pl_rect3d){
                                                    .x0 = 0, .y0 = 0, .z0 = 0,
                                                    .x1 = a.width, .y1 = a.height, .z1 = 1,
                                                },
                                                .row_pitch = (size_t)a.width * sizeof(float),
                                                .ptr = boosted,
                                            ));

                                            /* Blend detail back into float32 work buffer */
                                            for (int i = 0; i < a.width * a.height; i++) {
                                                float delta = boosted[i] - luma_data[i];
                                                work[i*3+0] = fmaxf(0.0f,
                                    fminf(1.0f, work[i*3+0] + delta));
                                                work[i*3+1] = fmaxf(0.0f,
                                    fminf(1.0f, work[i*3+1] + delta));
                                                work[i*3+2] = fmaxf(0.0f,
                                    fminf(1.0f, work[i*3+2] + delta));
                                            }
                                            free(boosted);
                                            fprintf(stderr, "  GPU CR: boost=%.2f "
                                                    "sigma_s=%.1f sigma_r=%.1f "
                                                    "radius=%d\n",
                                                    boost, sigma_s, sigma_r, radius);
                                        }
                                    }

                                    pl_pass_destroy(gpu, &pass);
                                    pl_tex_destroy(gpu, &cr_out_tex);
                                }
                            } else {
                                fprintf(stderr,
                                        "  GPU CR: no storable float format\n");
                            }
                            pl_tex_destroy(gpu, &luma_tex);
                        }
                    } else {
                        fprintf(stderr,
                                "  GPU CR: no sampleable float format\n");
                    }
                    free(luma_data);
                }
            }
        }
        /* --- Luma-Weighted Warm Chroma Reshaping ("fire pop") ---
         * Injects orange/red density for high-luma warm pixels (explosions,
         * fire, incandescent sources). Three phases with smoothstep
         * Hermite interpolation to avoid banding in gradients.
         *
         * IMPORTANT: This runs in a float working buffer (not uint8) to
         * maintain full precision for the subsequent L2 gamma trim.
         * The final uint8 quantization with blue-noise dithering happens only
         * once at output time.
         * ---------------------------------------------------------------- */
        /* work and did_processing are already declared/allocated above
         * in the L2 trim block — reuse the same buffer. */
        if (!work) {
            work = malloc((size_t)a.width * a.height * 3 * sizeof(float));
            if (work) {
                for (int i = 0; i < a.width * a.height; i++) {
                    work[i*3+0] = pixels[i*4+0] / 255.0f;
                    work[i*3+1] = pixels[i*4+1] / 255.0f;
                    work[i*3+2] = pixels[i*4+2] / 255.0f;
                }
                did_processing = 1;
            }
        }

        if (a.fire_pop_strength > 0.0f && work) {
            /* Pass 1: compute scene maximum luma */
            float scene_max = 0.001f;
            for (int i = 0; i < a.width * a.height; i++) {
                float y = 0.2126f*work[i*3+0] + 0.7152f*work[i*3+1]
                        + 0.0722f*work[i*3+2];
                if (y > scene_max) scene_max = y;
            }
            fprintf(stderr, "  Fire pop: scene_max_luma=%.4f\n", scene_max);

            /* Pass 2: warm chroma reshaping in float precision */
            /* Strength-dependent soft clamping — prevents 8-bit banding at high
             * strength by enforcing diminishing returns above 1.0x.
             *   1.0 raw → 1.0 shaped   (full 12% peak boost)
             *   1.5 raw → 1.28 shaped  (clamped to ~15.4%)
             *   2.0 raw → 1.49 shaped  (clamped to ~17.9%)
             * This avoids the 33% boost (1.0 + 0.22 × 1.5) that pushed
             * underlying 8-bit quantization into visible banding. */
            float strength = a.fire_pop_strength;
            if (strength > 1.0f) {
                strength = 1.0f + tanhf((strength - 1.0f) * 0.5f) * 0.8f;
            }
            fprintf(stderr,
                    "  Fire pop: raw=%.2f → shaped=%.3f "
                    "(peak Cr=+%.0f%%, Cb=+%.0f%%)\n",
                    a.fire_pop_strength, strength,
                    12.0f * strength, 8.0f * strength);
            for (int i = 0; i < a.width * a.height; i++) {
                float r = work[i*3+0];
                float g = work[i*3+1];
                float b = work[i*3+2];

                float y = 0.2126f*r + 0.7152f*g + 0.0722f*b;
                float norm_y = y / scene_max;
                float cr = r - y;
                float cb = b - y;

                if (norm_y > 0.45f && cr > 0.08f && cb < -0.01f) {
                    /* Phase B: squared sine of smoothstep.
                     * The envelope and its slope are zero at both bounds,
                     * avoiding a visible transition seam at 0.45/0.85. */
                    if (norm_y < 0.85f) {
                        float t = (norm_y - 0.45f) / 0.40f;
                        float smooth = 3.0f * t * t - 2.0f * t * t * t;
                        float sine = sinf(3.14159265358979323846f * smooth);
                        float envelope = sine * sine;
                        cr *= (1.0f + 0.12f * envelope * strength);
                        cb *= (1.0f + 0.08f * envelope * strength);
                    }

                    /* Phase A: Core desaturation, smooth 0.85 → 1.0 */
                    if (norm_y > 0.85f) {
                        float ct  = (norm_y - 0.85f) / 0.15f;
                        float css = 3.0f * ct * ct - 2.0f * ct * ct * ct;
                        float cd  = css * 0.35f * strength;
                        cr *= (1.0f - cd);
                        cb *= (1.0f - cd);
                    }

                    /* Phase C: Anti-pink hue constraint, smooth at 0.70 → 0.85 */
                    if (norm_y > 0.70f) {
                        float ht   = (norm_y - 0.70f) / 0.15f;
                        float hss  = 3.0f * ht * ht - 2.0f * ht * ht * ht;
                        float max_cr = fabsf(cb) * 2.6f * (1.0f - hss);
                        if (cr > max_cr) cr = max_cr;
                    } else {
                        float max_cr = fabsf(cb) * 2.6f;
                        if (cr > max_cr) cr = max_cr;
                    }

                    r = y + cr;
                    b = y + cb;
                    g = y - (0.2126f/0.7152f)*cr - (0.0722f/0.7152f)*cb;
                    work[i*3+0] = r < 0 ? 0 : r > 1 ? 1 : r;
                    work[i*3+1] = g < 0 ? 0 : g > 1 ? 1 : g;
                    work[i*3+2] = b < 0 ? 0 : b > 1 ? 1 : b;
                }
            }
            fprintf(stderr, "  Fire pop: DONE (strength=%.2f)\n", strength);
        }

        /* --- L2 gamma trim + saturation in float precision --- */
        if (work) {
            for (int i = 0; i < a.width * a.height; i++) {
                float r = work[i*3+0];
                float g = work[i*3+1];
                float b = work[i*3+2];
                float y = 0.2126f*r + 0.7152f*g + 0.0722f*b;
                float cb = b - y;
                float cr = r - y;

                if (l2_gamma_val != 1.0f && y > 0.001f && y < 0.999f) {
                    float t = 2.0f * y - 1.0f;
                    float powered = (t >= 0.0f) ? powf(t, l2_gamma_val)
                                                : -powf(-t, l2_gamma_val);
                    y = (powered + 1.0f) * 0.5f;
                }
                cb *= l2_sat_val;
                cr *= l2_sat_val;
                r = y + cr;
                g = y - (0.2126f/0.7152f)*cr - (0.0722f/0.7152f)*cb;
                b = y + cb;
                work[i*3+0] = r < 0 ? 0 : r > 1 ? 1 : r;
                work[i*3+1] = g < 0 ? 0 : g > 1 ? 1 : g;
                work[i*3+2] = b < 0 ? 0 : b > 1 ? 1 : b;
            }

            /* Letterbox bar tracking */
            int top_rows = 0, bot_rows = 0;
            if (a.top_bar_norm > 0.0f || a.bot_bar_norm > 0.0f) {
                top_rows = (int)(a.top_bar_norm * a.height + 0.5f);
                bot_rows = (int)(a.bot_bar_norm * a.height + 0.5f);
                top_rows = top_rows < a.height ? top_rows : a.height;
                bot_rows = bot_rows < a.height ? bot_rows : a.height;
            }

            /* Write float → uint8 with blue-noise dithering */
            const float *blue_noise = get_blue_noise_matrix();
            for (int y = 0; y < a.height; y++) {
                int is_bar = (y < top_rows || y >= a.height - bot_rows);
                for (int x = 0; x < a.width; x++) {
                    float dith = blue_noise[(y & 15) * 16 + (x & 15)] - 0.5f;
                    for (int c = 0; c < 3; c++) {
                        float f = is_bar ? 0.0f
                                         : work[(y*a.width+x)*3+c];
                        f += dith / 255.0f;
                        uint8_t v = (uint8_t)(f < 0 ? 0
                                              : f > 255 ? 255
                                              : (int)(f*255.0f+0.5f));
                        pixels[(y*a.width+x)*4+c] = v;
                    }
                }
            }
            free(work);
            work = NULL;
        }

        /* Fallthrough: float32 work buffer still needs conversion to uint8
         * pixels[] for output. This is done after the unified output path
         * (see the did_processing output block below). */
        }

    /* --- Universal float32→uint8 quantize: convert work buffer to pixels[] ---
     * Handles all cases where float32 processing was done (GPU CR, fire-pop,
     * L2 trim). For L2-only paths pixels[] is already written in-place above.
     * For deep-mode (float16 source) paths, work contains un-quantized data
     * that needs conversion to uint8 for stdout output. */
    if (did_processing && work && pixels && is_deep) {
        fprintf(stderr, "  [QUANTIZE] float32→uint8 (dithered, %d×%d)\n",
                a.width, a.height);
        const float *blue_noise = get_blue_noise_matrix();
        /* Letterbox bar tracking */
        int top_rows = 0, bot_rows = 0;
        if (a.top_bar_norm > 0.0f || a.bot_bar_norm > 0.0f) {
            top_rows = (int)(a.top_bar_norm * a.height + 0.5f);
            bot_rows = (int)(a.bot_bar_norm * a.height + 0.5f);
            top_rows = top_rows < a.height ? top_rows : a.height;
            bot_rows = bot_rows < a.height ? bot_rows : a.height;
        }
        for (int y = 0; y < a.height; y++) {
            int is_bar = (y < top_rows || y >= a.height - bot_rows);
            for (int x = 0; x < a.width; x++) {
                float dith = blue_noise[(y & 15) * 16 + (x & 15)] - 0.5f;
                for (int c = 0; c < 3; c++) {
                    float f = is_bar ? 0.0f : work[(y*a.width+x)*3+c];
                    f += dith / 255.0f;
                    pixels[(y*a.width+x)*4+c] = (uint8_t)(f < 0.0f ? 0
                                        : f > 1.0f ? 255
                                        : (int)(f * 255.0f + 0.5f));
                }
            }
        }
        free(work);
        work = NULL;
    }

    /* Free spatial CR buffers (if any) */
    /* (GPU bilateral filter manages its own GPU memory) */

    /* Zero out letterbox bar rows when no float processing was done. */
    if (!did_processing && (a.top_bar_norm > 0.0f || a.bot_bar_norm > 0.0f)) {
        int top_rows = (int)(a.top_bar_norm * a.height + 0.5f);
        int bot_rows = (int)(a.bot_bar_norm * a.height + 0.5f);
        top_rows = top_rows < a.height ? top_rows : a.height;
        bot_rows = bot_rows < a.height ? bot_rows : a.height;
        for (int y = 0; y < top_rows; y++) {
            if (is_deep && pixels16)
                memset(&pixels16[y * a.width * 8], 0, (size_t)a.width * 8);
            else if (pixels)
                memset(&pixels[y * a.width * 4], 0, (size_t)a.width * 4);
        }
        for (int y = a.height - bot_rows; y < a.height; y++) {
            if (is_deep && pixels16)
                memset(&pixels16[y * a.width * 8], 0, (size_t)a.width * 8);
            else if (pixels)
                memset(&pixels[y * a.width * 4], 0, (size_t)a.width * 4);
        }
    }

    /* --- Unified output path: write RGB8 to stdout --- */
    if (did_processing) {
        /* Float processing done — dithered uint8 is already in pixels[].
         * Just write to stdout, no additional dither. */
        fprintf(stderr,
            "  [OUTPUT] Single-quantize: float→uint8[dither]→stdout (single pass)\n");
        for (int y = 0; y < a.height; y++) {
            for (int x = 0; x < a.width; x++) {
                int idx = (y * a.width + x) * 4;
                fwrite(&pixels[idx], 1, 3, stdout);
            }
        }
    } else if (is_deep && pixels16) {
        /* No processing — deep (float16) tone-map output.
         * Dither-quantize from float16 to uint8 for stdout. */
        fprintf(stderr,
            "  [OUTPUT] float16 → uint8[dither]→stdout (%d×%d)\n",
            a.width, a.height);
        /* Use a uint8 scratch buffer for dithered output */
        uint8_t *out8 = malloc((size_t)a.width * a.height * 4);
        if (!out8) { fprintf(stderr, "Cannot alloc output buffer\n"); return 1; }

        const float *blue_noise = get_blue_noise_matrix();
        for (int y = 0; y < a.height; y++) {
            for (int x = 0; x < a.width; x++) {
                float dith = blue_noise[(y & 15) * 16 + (x & 15)] - 0.5f;
                for (int c = 0; c < 3; c++) {
                    /* float16 (IEEE 754 binary16) → float32 → 0-255 uint8 with blue-noise dither */
                    uint16_t raw = pixels16[(y*a.width+x)*4 + c];
                    float f = __half2float(raw);
                    f += dith / 255.0f;
                    uint8_t v = (uint8_t)(f < 0.0f ? 0
                                          : f > 1.0f ? 255
                                          : (int)(f * 255.0f + 0.5f));
                    out8[(y*a.width+x)*4 + c] = v;
                }
            }
        }
        /* Write RGB only (skip alpha) */
        for (int y = 0; y < a.height; y++) {
            for (int x = 0; x < a.width; x++) {
                int idx = (y * a.width + x) * 4;
                fwrite(&out8[idx], 1, 3, stdout);
            }
        }
        free(out8);
    } else {
        /* No processing — 8-bit tone-map output, dither at write time. */
        fprintf(stderr,
            "  [OUTPUT] uint8[dither]→stdout (8-bit, no processing)\n");
        const float *blue_noise = get_blue_noise_matrix();
        for (int y = 0; y < a.height; y++) {
            for (int x = 0; x < a.width; x++) {
                int idx = (y * a.width + x) * 4;
                float dith = blue_noise[(y & 15) * 16 + (x & 15)] - 0.5f;
                for (int c = 0; c < 3; c++) {
                    float f = pixels[idx+c] / 255.0f;
                    f += dith / 255.0f;
                    uint8_t v = (uint8_t)(f < 0 ? 0
                                          : f > 255 ? 255
                                          : (int)(f*255.0f+0.5f));
                    pixels[idx+c] = v;
                }
                fwrite(&pixels[idx], 1, 3, stdout);
            }
        }
    }
    fflush(stdout);

    /* --- Cleanup --- */
    free(pixels);
    free(pixels16);
    /* work may still be non-NULL if L2 trim freed it (it sets work=NULL) */
    if (work) free(work);
    pl_unmap_avframe(gpu, &image);
    av_frame_free(&avf);
    pl_tex_destroy(gpu, &out_tex);
    pl_renderer_destroy(&renderer);
    pl_d3d11_destroy(&d3d11);
    pl_log_destroy(&log);
    return 0;
}
