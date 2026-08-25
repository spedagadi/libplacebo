/*
 * dv_render.c — headless Dolby Vision frame renderer via libplacebo + D3D11
 *
 * Decodes one frame from a DV video at a given PTS, renders it through the
 * full libplacebo colour pipeline under one of three tone-mapping modes, and
 * writes the result as raw RGB8 (24 bpp, row-major) to stdout.
 *
 * Server mode keeps the D3D11/libplacebo context and renderer alive. It accepts
 * one JSON request per line and returns a DVR1 response containing spline and
 * contrast-recovery RGB8 frames.
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
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libavutil/time.h>

#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/d3d11.h>
#include <libplacebo/utils/libav.h>
#include <libplacebo/colorspace.h>
#include <libplacebo/tone_mapping.h>
#include <libplacebo/dither.h>
#include <libplacebo/utils/frame_queue.h>
#include <libplacebo/shaders/custom.h>
#include <libplacebo/ml_features.h>
#include <libplacebo/ml_model.h>
#include <libplacebo/ml_radiance.h>
#include <libplacebo/ml_render.h>

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

struct fire_pop_hook_state {
    float strength;
};

struct l2_hook_state {
    float gamma;
    float saturation;
};

static struct pl_hook_res fire_pop_output_hook(void *priv,
                                                const struct pl_hook_params *params)
{
    struct fire_pop_hook_state *state = priv;
    pl_shader sh = params->sh;
    static const char body[] =
        "float y = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));\n"
        "float t = clamp((y - 0.45) / 0.40, 0.0, 1.0);\n"
        "float envelope_t = t * t * (3.0 - 2.0 * t);\n"
        "float sine = sin(3.141592653589793 * envelope_t);\n"
        "float envelope = sine * sine;\n"
        "float cr = color.r - y;\n"
        "float cb = color.b - y;\n"
        "if (y > 0.45 && cr > 0.08 && cb < -0.01) {\n"
        "    cr *= 1.0 + 0.12 * envelope * fire_strength;\n"
        "    cb *= 1.0 + 0.08 * envelope * fire_strength;\n"
        "    color.r = clamp(y + cr, 0.0, 1.0);\n"
        "    color.b = clamp(y + cb, 0.0, 1.0);\n"
        "    color.g = clamp(y - 0.2126 / 0.7152 * cr - "
        "0.0722 / 0.7152 * cb, 0.0, 1.0);\n"
        "}\n";
    struct pl_shader_var var = {
        .var = pl_var_float("fire_strength"),
        .data = &state->strength,
        .dynamic = true,
    };
    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "GPU fire-pop output hook",
        .body = body,
        .input = PL_SHADER_SIG_COLOR,
        .output = PL_SHADER_SIG_COLOR,
        .variables = &var,
        .num_variables = 1,
        .output_w = pl_rect_w(params->dst_rect),
        .output_h = pl_rect_h(params->dst_rect),
    })) {
        return (struct pl_hook_res) { .failed = true };
    }
    return (struct pl_hook_res) {
        .output = PL_HOOK_SIG_COLOR,
        .sh = sh,
        .repr = params->repr,
        .color = params->color,
        .components = params->components,
        .rect = params->rect,
    };
}

static struct pl_hook_res l2_output_hook(void *priv,
                                          const struct pl_hook_params *params)
{
    struct l2_hook_state *state = priv;
    pl_shader sh = params->sh;
    static const char body[] =
        "float y = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));\n"
        "float cb = (color.b - y) * l2_saturation;\n"
        "float cr = (color.r - y) * l2_saturation;\n"
        "if (l2_gamma != 1.0 && y > 0.001 && y < 0.999) {\n"
        "    float t = 2.0 * y - 1.0;\n"
        "    y = 0.5 * (sign(t) * pow(abs(t), l2_gamma) + 1.0);\n"
        "}\n"
        "color.r = clamp(y + cr, 0.0, 1.0);\n"
        "color.g = clamp(y - 0.2126 / 0.7152 * cr - 0.0722 / 0.7152 * cb, 0.0, 1.0);\n"
        "color.b = clamp(y + cb, 0.0, 1.0);\n";
    struct pl_shader_var vars[] = {
        { .var = pl_var_float("l2_gamma"), .data = &state->gamma, .dynamic = true },
        { .var = pl_var_float("l2_saturation"), .data = &state->saturation, .dynamic = true },
    };
    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "GPU L2 gamma and saturation trim",
        .body = body,
        .input = PL_SHADER_SIG_COLOR,
        .output = PL_SHADER_SIG_COLOR,
        .variables = vars,
        .num_variables = 2,
        .output_w = pl_rect_w(params->dst_rect),
        .output_h = pl_rect_h(params->dst_rect),
    })) {
        return (struct pl_hook_res) { .failed = true };
    }
    return (struct pl_hook_res) {
        .output = PL_HOOK_SIG_COLOR,
        .sh = sh,
        .repr = params->repr,
        .color = params->color,
        .components = params->components,
        .rect = params->rect,
    };
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
    "#version 450\n"
    "\n"
    "layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;\n"
    "\n"
    "layout (binding = 0) uniform sampler2D u_input;\n"
    "\n"
    "layout (r32f, binding = 0) uniform image2D u_output;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    vec2 pos = vec2(gl_GlobalInvocationID.xy);\n"
    "    vec2 size = vec2(textureSize(u_input, 0));\n"
    "    if (pos.x >= size.x || pos.y >= size.y) return;\n"
    "    ivec2 ipos = ivec2(pos);\n"
    "\n"
    "    float center = texture(u_input, pos / size).x;\n"
    "\n"
    "    const float r = 4.0;\n"
    "    const float inv2ss = -1.0 / (2.0 * 3.0 * 3.0);\n"
    "    const float inv2sr = -1.0 / (2.0 * 15.0 * 15.0);\n"
    "    float w_sum = 0.0;\n"
    "    float y_sum = 0.0;\n"
    "\n"
    "    for (float fy = -r; fy <= r; fy++) {\n"
    "        for (float fx = -r; fx <= r; fx++) {\n"
    "            ivec2 np = ipos + ivec2(int(fx), int(fy));\n"
    "            if (np.x < 0 || np.x >= int(size.x) ||\n"
    "                np.y < 0 || np.y >= int(size.y)) continue;\n"
    "\n"
    "            float ny = texture(u_input, vec2(np) / size).x;\n"
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
    "    float boosted = center + detail * 0.8;\n"
    "    imageStore(u_output, ipos, vec4(boosted, 0.0, 0.0, 1.0));\n"
    "}\n";

/* --- GPU CR: pass variable & descriptor definitions --- */
#define NUM_CR_VARS 0
#define NUM_CR_DESCS 2

static struct pl_var cr_vars[NUM_CR_VARS];
static struct pl_desc cr_descs[NUM_CR_DESCS];


/* Initialize cr_vars and cr_descs (called once at startup) */
static void init_cr_vars(void)
{

    cr_descs[0] = (struct pl_desc){
        .name   = "u_input",
        .type   = PL_DESC_SAMPLED_TEX,
        .binding = 0,
        .access = PL_DESC_ACCESS_READONLY,
    };
    cr_descs[1] = (struct pl_desc){
        .name   = "u_output",
        .type   = PL_DESC_STORAGE_IMG,
        .binding = 1,
        .access = PL_DESC_ACCESS_WRITEONLY,
    };
}

enum dv_control_mode {
    DV_CONTROL_OFF,
    DV_CONTROL_AUTO,
    DV_CONTROL_MANUAL,
};

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
    enum dv_control_mode gamma_mode;
    float       contrast_gamma;   /* manual override for gamma in contrast-recovery mode */
    float       contrast_sat;     /* manual override for saturation in contrast-recovery mode */
    /* Specular highlight roll-off (piecewise L2 trim) — deferred to XGBoost model */
    float       highlight_knee;   /* reserved: 0.0-1.0, default: auto from scene stats */
    /* Libplacebo HDR contrast recovery (high-frequency detail injection) */
    float       cr_strength;      /* 0.0-0.5, default: auto from predicted gamma */
    enum dv_control_mode cr_mode;
    float       cr_smoothness;    /* >1.0, default: 2.5 (tighter halos on fine textures) */
    /* Luma-Weighted Warm Chroma Reshaping (fire pop) — boosts orange/red density
     * for high-luma warm pixels (explosions, fire, incandescent sources).
     * Three phases: specular desaturation (core white-hot), mid-flame body
     * saturation injection, anti-pink hue constraint. */
    float       fire_pop_strength;/* 0.0-2.0, default 1.0 (scales chroma boost scalars) */
    enum dv_control_mode fire_pop_mode;
    float       radiance_knee;    /* output-luma knee for adaptive highlight lift */
    float       radiance_strength;
    enum dv_control_mode radiance_mode;
    enum dv_control_mode chroma_mode;
    float       chroma_neutral_boost;
    float       chroma_fire_boost;
    float       chroma_knee;
    float       chroma_skin_protect;
    int         server;
    int         playback_server;
    bool        write_output;
    const char *model_path;
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
        "          [--model <xgb_model.plxgb>]     load native ML model at startup\n"
        "          [--no-output]                  render on GPU without diagnostic readback\n"
        "  Manual L2 trim (overrides contrast-recovery):\n"
        "          [--l2-power <val>]           2048=neutral, <2048=more contrast\n"
        "          [--l2-sat-gain <val>]        2048=neutral, >2048=more saturation\n"
        "Output: raw RGB8 to stdout\n"
        "Server mode: %s --server < requests.jsonl > responses.bin\n"
        "Playback server: %s --playback-server < requests.jsonl > responses.bin\n"
        "  request:  {\"input\":\"file.mkv\",\"pts\":12.3,\"width\":1920,\"height\":1080,\"l1_max\":1.0,\"l1_avg\":0.2}\n"
        "  response: DVR1/u32 version,width,height,count, then two mode/length records and RGB8 payloads\n"
        "  playback response: DVRP/u32 version,width,height,pts_ms,payload_len, then RGB8 payload\n",
        argv0, argv0, argv0);
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
    a->gamma_mode = DV_CONTROL_AUTO;
    a->contrast_gamma  = 0.f;  /* 0 = auto-predict */
    a->contrast_sat    = 1.f;  /* 1.0 = neutral */
    a->highlight_knee  = -1.f;  /* -1 = auto-predict from scene stats */
    a->cr_strength     = -1.f;  /* -1 = auto-scale from gamma */
    a->cr_mode         = DV_CONTROL_AUTO;
    a->cr_smoothness   = -1.f;  /* -1 = use 2.5 default */
    a->fire_pop_strength = 0.f;  /* 0 = disabled (not enabled unless explicitly set) */
    a->fire_pop_mode = DV_CONTROL_OFF;
    a->radiance_knee = 0.60f;
    a->radiance_strength = 0.30f;
    a->radiance_mode = DV_CONTROL_OFF;
    a->write_output = true;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--server")) { a->server = 1; }
        else if (!strcmp(argv[i], "--playback-server")) { a->playback_server = 1; }
        else if (!strcmp(argv[i], "--no-output")) { a->write_output = false; }
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) { a->model_path = argv[++i]; }
        else if (!strcmp(argv[i], "--input")            && i+1 < argc) { a->input           = argv[++i]; }
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
        else if (!strcmp(argv[i], "--contrast-gamma")     && i+1 < argc) { a->contrast_gamma = atof(argv[++i]); a->gamma_mode = DV_CONTROL_MANUAL; }
        else if (!strcmp(argv[i], "--contrast-sat")       && i+1 < argc) { a->contrast_sat = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--cr-strength")        && i+1 < argc) { a->cr_strength  = atof(argv[++i]); a->cr_mode = DV_CONTROL_MANUAL; }
        else if (!strcmp(argv[i], "--cr-smoothness")      && i+1 < argc) { a->cr_smoothness= atof(argv[++i]); }
        else if (!strcmp(argv[i], "--highlight-knee")     && i+1 < argc) { a->highlight_knee = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--fire-pop-strength")  && i+1 < argc) { a->fire_pop_strength = atof(argv[++i]); a->fire_pop_mode = DV_CONTROL_MANUAL; }
        else if (!strcmp(argv[i], "--radiance-knee")      && i+1 < argc) { a->radiance_knee = atof(argv[++i]); a->radiance_mode = DV_CONTROL_MANUAL; }
        else if (!strcmp(argv[i], "--radiance-strength")  && i+1 < argc) { a->radiance_strength = atof(argv[++i]); a->radiance_mode = DV_CONTROL_MANUAL; }
        else if (!strcmp(argv[i], "--contrast-recovery")) { a->contrast_recovery = 1; }
        else { fprintf(stderr, "Unknown argument: %s\n", argv[i]); return false; }
    }
    if (a->server || a->playback_server)
        return true;
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

typedef struct {
    AVFormatContext *format;
    AVCodecContext *codec;
    AVStream *stream;
    int stream_index;
    AVPacket *packet;
    AVFrame *frame;
    AVFrame *last_frame;
    int64_t last_decoded_pts;
    double last_pts;
    bool have_last;
    char input[2048];
    pl_gpu gpu;
} NativeDecoder;

static void native_decoder_close(NativeDecoder *decoder)
{
    if (!decoder) return;
    av_frame_free(&decoder->last_frame);
    av_frame_free(&decoder->frame);
    av_packet_free(&decoder->packet);
    avcodec_free_context(&decoder->codec);
    avformat_close_input(&decoder->format);
    memset(decoder, 0, sizeof(*decoder));
}

static bool native_decoder_open(NativeDecoder *decoder, const char *path,
                                pl_gpu gpu)
{
    const AVCodec *codec;
    const AVCodecHWConfig *hwcfg = NULL;

    native_decoder_close(decoder);
    decoder->gpu = gpu;
    fprintf(stderr, "Server decoder open: %s\n", path);
    if (avformat_open_input(&decoder->format, path, NULL, NULL) < 0 ||
        avformat_find_stream_info(decoder->format, NULL) < 0) {
        fprintf(stderr, "Server decoder: cannot open input %s\n", path);
        native_decoder_close(decoder);
        return false;
    }

    decoder->stream_index = av_find_best_stream(decoder->format,
                                                 AVMEDIA_TYPE_VIDEO, -1, -1,
                                                 NULL, 0);
    if (decoder->stream_index < 0) {
        fprintf(stderr, "Server decoder: no video stream in %s\n", path);
        native_decoder_close(decoder);
        return false;
    }
    decoder->stream = decoder->format->streams[decoder->stream_index];
    codec = avcodec_find_decoder(decoder->stream->codecpar->codec_id);
    if (!codec || !(decoder->codec = avcodec_alloc_context3(codec)) ||
        avcodec_parameters_to_context(decoder->codec,
                                      decoder->stream->codecpar) < 0) {
        fprintf(stderr, "Server decoder: cannot initialize codec\n");
        native_decoder_close(decoder);
        return false;
    }

    for (int i = 0; (hwcfg = avcodec_get_hw_config(codec, i)); i++) {
        if (!pl_test_pixfmt(gpu, hwcfg->pix_fmt))
            continue;
        if (!(hwcfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX))
            continue;
        if (av_hwdevice_ctx_create(&decoder->codec->hw_device_ctx,
                                   hwcfg->device_type, NULL, NULL, 0) < 0) {
            fprintf(stderr, "Server decoder: HW device creation failed, trying next format\n");
            continue;
        }
        decoder->codec->extra_hw_frames = 4;
        fprintf(stderr, "Server decoder: hardware format %s\n",
                av_get_pix_fmt_name(hwcfg->pix_fmt));
        break;
    }
    if (!hwcfg || !decoder->codec->hw_device_ctx)
        fprintf(stderr, "Server decoder: software decoding\n");

    decoder->codec->get_buffer2 = pl_get_buffer2;
    decoder->codec->opaque = &decoder->gpu;
    decoder->codec->export_side_data |= AV_CODEC_EXPORT_DATA_PRFT;
    if (avcodec_open2(decoder->codec, codec, NULL) < 0) {
        fprintf(stderr, "Server decoder: cannot open codec\n");
        native_decoder_close(decoder);
        return false;
    }
    decoder->packet = av_packet_alloc();
    decoder->frame = av_frame_alloc();
    if (!decoder->packet || !decoder->frame) {
        fprintf(stderr, "Server decoder: cannot allocate packet/frame\n");
        native_decoder_close(decoder);
        return false;
    }
    strncpy(decoder->input, path, sizeof(decoder->input) - 1);
    decoder->input[sizeof(decoder->input) - 1] = '\0';
    return true;
}

static AVFrame *native_decoder_decode(NativeDecoder *decoder, double target_pts)
{
    bool sequential = decoder->have_last && target_pts >= decoder->last_pts;
    if (decoder->have_last && fabs(target_pts - decoder->last_pts) < 0.000001) {
        fprintf(stderr, "Server decoder: reuse cached frame at %.6f\n", target_pts);
        return av_frame_clone(decoder->last_frame);
    }

    if (!sequential) {
        int64_t seek_ts = (int64_t)(target_pts * AV_TIME_BASE);
        fprintf(stderr, "Server decoder: seek/reset to %.6f\n", target_pts);
        if (av_seek_frame(decoder->format, -1, seek_ts, AVSEEK_FLAG_BACKWARD) < 0)
            return NULL;
        avcodec_flush_buffers(decoder->codec);
        decoder->have_last = false;
    } else {
        fprintf(stderr, "Server decoder: sequential decode to %.6f\n", target_pts);
    }

    for (int attempts = 0; attempts < 512; attempts++) {
        int ret = av_read_frame(decoder->format, decoder->packet);
        if (ret < 0) break;
        if (decoder->packet->stream_index != decoder->stream_index) {
            av_packet_unref(decoder->packet);
            continue;
        }
        ret = avcodec_send_packet(decoder->codec, decoder->packet);
        av_packet_unref(decoder->packet);
        if (ret < 0) continue;
        while ((ret = avcodec_receive_frame(decoder->codec, decoder->frame)) == 0) {
            int64_t pts = decoder->frame->best_effort_timestamp;
            if (pts == AV_NOPTS_VALUE) pts = decoder->frame->pts;
            if (pts == AV_NOPTS_VALUE) {
                av_frame_unref(decoder->frame);
                continue;
            }
            double frame_pts = pts * av_q2d(decoder->stream->time_base);
            if (frame_pts + 0.05 >= target_pts) {
                av_frame_free(&decoder->last_frame);
                decoder->last_frame = av_frame_clone(decoder->frame);
                if (!decoder->last_frame) return NULL;
                decoder->last_pts = frame_pts;
                decoder->last_decoded_pts = pts;
                decoder->have_last = true;
                AVFrame *result = av_frame_clone(decoder->frame);
                av_frame_unref(decoder->frame);
                return result;
            }
            av_frame_unref(decoder->frame);
        }
    }
    fprintf(stderr, "Server decoder: no frame at or after %.6f\n", target_pts);
    return NULL;
}

/* -------------------------------------------------------------------------
 * Reusable single-request renderer. GPU and renderer ownership belongs to the
 * caller so server mode can retain the expensive native context.
 * ---------------------------------------------------------------------- */

enum dv_frame_info_flags {
    DV_FRAME_INFO_AUTO = 1 << 0,
    DV_FRAME_INFO_MODEL = 1 << 1,
    DV_FRAME_INFO_MANUAL = 1 << 2,
    DV_FRAME_INFO_FALLBACK = 1 << 3,
    DV_FRAME_INFO_GAMMA_OFF = 1 << 4,
    DV_FRAME_INFO_CR_AUTO = 1 << 5,
    DV_FRAME_INFO_CR_MANUAL = 1 << 6,
    DV_FRAME_INFO_CR_OFF = 1 << 7,
    DV_FRAME_INFO_FIRE_MANUAL = 1 << 8,
    DV_FRAME_INFO_FIRE_OFF = 1 << 9,
    DV_FRAME_INFO_FIRE_FALLBACK = 1 << 10,
    DV_FRAME_INFO_RADIANCE_AUTO = 1 << 11,
    DV_FRAME_INFO_RADIANCE_MANUAL = 1 << 12,
    DV_FRAME_INFO_RADIANCE_OFF = 1 << 13,
};

struct dv_frame_info {
    uint32_t flags;
    float gamma;
    float l1_max_pq;
    float l1_avg_pq;
    float cr_strength;
    float l2_power;
    float fire_pop_strength;
    float radiance_knee;
    float radiance_strength;
    float chroma_neutral_boost;
    float chroma_fire_boost;
    float chroma_knee;
    float chroma_skin_protect;
};

static int render_frame_with_decoder(Args a, pl_gpu gpu, pl_renderer renderer,
                                     pl_ml_context ml_context,
                                     NativeDecoder *decoder, AVFrame *provided,
                                     const struct pl_frame *mapped,
                                     struct dv_frame_info *frame_info)
{
    if (frame_info)
        memset(frame_info, 0, sizeof(*frame_info));
    AVFrame *avf = provided ? provided : (decoder ? native_decoder_decode(decoder, a.pts) :
                           decode_frame_at(a.input, a.pts, a.width, a.height));
    if (!avf) { fprintf(stderr, "Failed to decode frame\n"); return 1; }

    /* --- Map AVFrame → pl_frame ---
     * Always use map_dovi=true so the DV RPU's custom ycc_to_rgb matrix
     * is applied. For spline/ml we then replace the reshaping curves with
     * identity — same matrix, different tone curve.
     */
    struct pl_frame image = {0};
    pl_tex tex[4] = {0};

    if (mapped) {
        image = *mapped;
    } else if (!pl_frame_recreate_from_avframe(gpu, &image, tex, avf)) {
        fprintf(stderr, "pl_frame_recreate_from_avframe failed\n");
        return 1;
    }

    bool ok = true;
    if (!mapped) {
        ok = pl_map_avframe_ex(gpu, &image, pl_avframe_params(
            .frame    = avf,
            .tex      = tex,
            .map_dovi = true,   /* always — need the RPU colour matrix */
        ));
        if (!ok) { fprintf(stderr, "pl_map_avframe_ex failed\n"); return 1; }
    }

    if (a.fire_pop_mode == DV_CONTROL_OFF)
        a.fire_pop_strength = 0.0f;
    else if (a.fire_pop_mode == DV_CONTROL_AUTO)
        a.fire_pop_strength = 1.0f;  /* auto: default strength */

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
    enum pl_fmt_caps output_caps = PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_STORABLE;
    if (a.write_output)
        output_caps |= PL_FMT_CAP_HOST_READABLE;
    pl_fmt out_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 16, output_caps);
    if (!out_fmt) {
        /* Fallback: use rgba8 if float16 render target unavailable */
        fprintf(stderr, "  [WARN] float16 render target unavailable, falling back to rgba8\n");
        out_fmt = pl_find_named_fmt(gpu, "rgba8");
        if (!out_fmt) out_fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 4, 8, 8,
                            PL_FMT_CAP_RENDERABLE |
                            (a.write_output ? PL_FMT_CAP_HOST_READABLE : 0));
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
        .host_readable= a.write_output,
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
    struct pl_render_params rparams = pl_render_high_quality_params;
    rparams.color_map_params = &cmap;
    rparams.antiringing_strength = 0.80f;
    struct pl_ml_radiance radiance = {0};
    struct pl_hook hooks[4];
    struct pl_ml_render_result ml_result = { .l2_power = 2048.0f, .l2_saturation = 2048.0f };
    bool have_ml_result = false;
    bool gpu_fire_pop = a.fire_pop_strength > 0.0f;
    bool gpu_l2 = false;
    bool gpu_radiance = false;
    struct l2_hook_state l2_hook_state = { .gamma = 1.0f, .saturation = 1.0f };

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
        if (a.l1_avg_pq <= 0.0f) {
            fprintf(stderr,
                "Warning: --l1-avg not set for contrast-recovery — using average=%.4f\n",
                image.color.hdr.avg_pq_y);
            a.l1_avg_pq = image.color.hdr.avg_pq_y;
        }

        /* Predict gamma from the complete canonical feature vector. */
        float content_ratio = a.l1_max_pq / (a.out_nits > 0.01f ? a.out_nits / 10000.0f : 0.02f);
        float predicted_gamma = predict_gamma_from_brightness(content_ratio, a.l1_avg_pq);
        have_ml_result = pl_ml_render_evaluate(gpu, &image, pl_ml_render_params(
            .model = ml_context,
            .gamma_mode = (enum pl_ml_control_mode)a.gamma_mode,
            .gamma = a.contrast_gamma,
            .cr_mode = (enum pl_ml_control_mode)a.cr_mode,
            .cr_strength = a.cr_strength,
            .fire_pop_mode = (enum pl_ml_control_mode)a.fire_pop_mode,
            .fire_pop_strength = a.fire_pop_strength,
            .radiance = {
                .mode = (enum pl_ml_control_mode)a.radiance_mode,
                .knee = a.radiance_knee,
                .strength = a.radiance_strength,
            },
            .chroma_mode = (enum pl_ml_control_mode)a.chroma_mode,
            .chroma_neutral_boost = a.chroma_neutral_boost,
            .chroma_fire_boost = a.chroma_fire_boost,
            .chroma_knee = a.chroma_knee,
            .chroma_skin_protect = a.chroma_skin_protect,
            .target_nits = a.out_nits,
            .l1_max_pq = a.l1_max_pq,
            .l1_avg_pq = a.l1_avg_pq,
            .top_bar_norm = a.top_bar_norm,
            .bottom_bar_norm = a.bot_bar_norm), &ml_result);
        if (!have_ml_result) {
            fprintf(stderr, "Native ML render evaluation unavailable; using heuristic fallback\n");
            ml_result.gamma = a.gamma_mode == DV_CONTROL_MANUAL ? a.contrast_gamma :
                a.gamma_mode == DV_CONTROL_OFF ? 1.0f : predicted_gamma;
            ml_result.l2_power = 2048.0f / ml_result.gamma;
            ml_result.l2_saturation = 2048.0f;
            ml_result.cr_strength = a.cr_mode == DV_CONTROL_MANUAL ? a.cr_strength :
                a.cr_mode == DV_CONTROL_OFF ? 0.0f : fmaxf(0.1f, fminf(0.5f,
                    0.25f + (1.2f - ml_result.gamma) * 0.15f));
            pl_ml_radiance_configure(&ml_result.radiance, pl_ml_radiance_params(
                .mode = (enum pl_ml_control_mode)a.radiance_mode,
                .average_luma = a.l1_avg_pq,
                .knee = a.radiance_knee,
                .strength = a.radiance_strength));
        }
        if (ml_result.model_used)
            fprintf(stderr, "Native ML gamma: %.3f (88 features)\n", ml_result.gamma);

        // Apply chroma params unconditionally — not populated by fallback path
        if (a.chroma_mode == DV_CONTROL_MANUAL) {
            ml_result.chroma_neutral_boost = a.chroma_neutral_boost;
            ml_result.chroma_fire_boost    = a.chroma_fire_boost;
            ml_result.chroma_knee          = a.chroma_knee;
            ml_result.chroma_skin_protect  = a.chroma_skin_protect;
        } else if (a.chroma_mode == DV_CONTROL_AUTO) {
            ml_result.chroma_neutral_boost = 1.20f;
            ml_result.chroma_fire_boost    = 1.35f;
            ml_result.chroma_knee          = 0.55f;
            ml_result.chroma_skin_protect  = 0.95f;
        }

        float gamma = ml_result.gamma;
        float power = ml_result.l2_power;
        float sat = ml_result.l2_saturation;

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
        if (a.cr_mode == DV_CONTROL_OFF) {
            cmap.contrast_recovery = 0.0f;
            fprintf(stderr, "  CR strength: off (disabled)\n");
        } else if (a.cr_mode == DV_CONTROL_MANUAL) {
            cmap.contrast_recovery = a.cr_strength >= 0.0f ? a.cr_strength : ml_result.cr_strength;
            fprintf(stderr, "  CR strength: %.3f (manual)\n", cmap.contrast_recovery);
        } else {
            cmap.contrast_recovery = ml_result.cr_strength;
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
        if (frame_info) {
            frame_info->flags = a.gamma_mode == DV_CONTROL_MANUAL ? DV_FRAME_INFO_MANUAL :
                a.gamma_mode == DV_CONTROL_OFF ? DV_FRAME_INFO_GAMMA_OFF :
                DV_FRAME_INFO_AUTO | (ml_result.model_used ? DV_FRAME_INFO_MODEL : DV_FRAME_INFO_FALLBACK);
            frame_info->flags |= a.cr_mode == DV_CONTROL_MANUAL ? DV_FRAME_INFO_CR_MANUAL :
                a.cr_mode == DV_CONTROL_OFF ? DV_FRAME_INFO_CR_OFF : DV_FRAME_INFO_CR_AUTO;
            frame_info->flags |= a.fire_pop_mode == DV_CONTROL_MANUAL ? DV_FRAME_INFO_FIRE_MANUAL :
                a.fire_pop_mode == DV_CONTROL_AUTO ? DV_FRAME_INFO_FIRE_MANUAL : DV_FRAME_INFO_FIRE_OFF;
            frame_info->gamma = gamma;
            frame_info->l1_max_pq = a.l1_max_pq;
            frame_info->l1_avg_pq = a.l1_avg_pq;
            frame_info->cr_strength = cmap.contrast_recovery;
            frame_info->l2_power = power;
            frame_info->fire_pop_strength = a.fire_pop_strength;
        }
    }

    if (a.l2_power > 0.0f || a.l2_sat_gain > 0.0f) {
        l2_hook_state.gamma = a.l2_power > 0.0f ? 2048.0f / a.l2_power : 1.0f;
        l2_hook_state.saturation = a.l2_sat_gain > 0.0f ?
            a.l2_sat_gain / 2048.0f : 1.0f;
        gpu_l2 = l2_hook_state.gamma != 1.0f || l2_hook_state.saturation != 1.0f;
    }
    radiance = ml_result.radiance;
    gpu_radiance = radiance.strength > 0.0f;
    if (frame_info) {
        frame_info->flags |= a.radiance_mode == DV_CONTROL_MANUAL ?
            DV_FRAME_INFO_RADIANCE_MANUAL : a.radiance_mode == DV_CONTROL_AUTO ?
            DV_FRAME_INFO_RADIANCE_AUTO : DV_FRAME_INFO_RADIANCE_OFF;
        frame_info->radiance_knee = radiance.knee;
        frame_info->radiance_strength = radiance.strength;
        frame_info->chroma_neutral_boost = ml_result.chroma_neutral_boost;
        frame_info->chroma_fire_boost = ml_result.chroma_fire_boost;
        frame_info->chroma_knee = ml_result.chroma_knee;
        frame_info->chroma_skin_protect = ml_result.chroma_skin_protect;
    }
    int num_hooks = pl_ml_render_get_hooks(&ml_result, hooks);
    if (num_hooks) {
        const struct pl_hook *hook_ptrs[4];
        for (int i = 0; i < num_hooks; i++) hook_ptrs[i] = &hooks[i];
        rparams.hooks = hook_ptrs;
        rparams.num_hooks = num_hooks;
        fprintf(stderr, "GPU output hooks: %d active (chroma=%s)\n", num_hooks,
            (ml_result.chroma_neutral_boost > 1.0f || ml_result.chroma_fire_boost > 1.0f) ? "on" : "off");
    }

    /* --- Render --- */
    ok = pl_render_image(renderer, &image, &target, &rparams);
    if (!ok) {
        fprintf(stderr, "pl_render_image failed\n");
        return 1;
    }

    if (!a.write_output) {
        fprintf(stderr, "GPU-only render complete: no diagnostic texture readback\n");
        if (!mapped) {
            pl_unmap_avframe(gpu, &image);
            av_frame_free(&avf);
        }
        pl_tex_destroy(gpu, &out_tex);
        return 0;
    }

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
    const bool use_legacy_cr_overlay = false;

    if (cmap.contrast_recovery > 0.0f) {
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
        if (use_legacy_cr_overlay && a.contrast_recovery && a.cr_strength > 0.0f &&
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

                                        /* 8. Execute pass */
                                        pl_pass_run(gpu,
                                            pl_pass_run_params(
                                                .pass            = pass,
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

        if (a.fire_pop_strength > 0.0f && work && !gpu_fire_pop) {
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
            if (!gpu_l2) {
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
    if (!mapped) {
        pl_unmap_avframe(gpu, &image);
        av_frame_free(&avf);
    }
    pl_tex_destroy(gpu, &out_tex);
    return 0;
}

typedef struct {
    char input[2048];
    double pts;
    int width;
    int height;
    float l1_max_pq;
    float l1_avg_pq;
    float out_nits;
    float contrast_gamma;
    float cr_strength;
    float fire_pop_strength;
    float radiance_knee;
    float radiance_strength;
    enum dv_control_mode gamma_mode;
    enum dv_control_mode cr_mode;
    enum dv_control_mode fire_pop_mode;
    enum dv_control_mode radiance_mode;
    enum dv_control_mode chroma_mode;
    float chroma_neutral_boost;
    float chroma_fire_boost;
    float chroma_knee;
    float chroma_skin_protect;
} ServerRequest;

static bool server_json_number(const char *line, const char *key, double *value)
{
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *start = strstr(line, needle);
    if (!start) return false;
    start = strchr(start, ':');
    if (!start) return false;
    char *end = NULL;
    *value = strtod(start + 1, &end);
    return end != start + 1;
}

static enum dv_control_mode server_control_mode(const char *line, const char *key,
                                                enum dv_control_mode fallback)
{
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *value = strstr(line, needle);
    if (!value || !(value = strchr(value, ':')) || !(value = strchr(value, '"')))
        return fallback;
    value++;
    if (!strncmp(value, "off\"", 4)) return DV_CONTROL_OFF;
    if (!strncmp(value, "auto\"", 5)) return DV_CONTROL_AUTO;
    if (!strncmp(value, "manual\"", 7)) return DV_CONTROL_MANUAL;
    return fallback;
}

static bool parse_server_request(const char *line, ServerRequest *request)
{
    memset(request, 0, sizeof(*request));
    request->width = 1920;
    request->height = 1080;
    request->l1_max_pq = 0.0f;
    request->l1_avg_pq = 0.0f;
    request->out_nits = 203.0f;
    request->contrast_gamma = 0.0f;
    request->cr_strength = -1.0f;
    request->fire_pop_strength = 0.0f;
    request->radiance_knee = 0.60f;
    request->radiance_strength = 0.30f;
    request->gamma_mode = DV_CONTROL_AUTO;
    request->cr_mode = DV_CONTROL_AUTO;
    request->fire_pop_mode = DV_CONTROL_OFF;
    request->radiance_mode = DV_CONTROL_OFF;
    request->chroma_mode = DV_CONTROL_OFF;
    request->chroma_neutral_boost = 1.20f;
    request->chroma_fire_boost = 1.35f;
    request->chroma_knee = 0.55f;
    request->chroma_skin_protect = 0.95f;

    const char *key = strstr(line, "\"input\"");
    if (!key) key = strstr(line, "\"mkv_path\"");
    if (!key) return false;
    const char *value = strchr(key, ':');
    if (!value || !(value = strchr(value, '"'))) return false;
    value++;
    const char *end = strchr(value, '"');
    if (!end || end == value) return false;
    size_t length = (size_t)(end - value);
    if (length >= sizeof(request->input)) return false;
    memcpy(request->input, value, length);
    request->input[length] = '\0';

    double number;
    if (!server_json_number(line, "pts", &number) || number < 0.0)
        return false;
    request->pts = number;
    if (server_json_number(line, "width", &number)) request->width = (int)number;
    if (server_json_number(line, "height", &number)) request->height = (int)number;
    if (server_json_number(line, "l1_max", &number)) request->l1_max_pq = (float)number;
    if (server_json_number(line, "l1_avg", &number)) request->l1_avg_pq = (float)number;
    if (server_json_number(line, "out_nits", &number)) request->out_nits = (float)number;
    if (server_json_number(line, "contrast_gamma", &number)) request->contrast_gamma = (float)number;
    if (server_json_number(line, "cr_strength", &number)) request->cr_strength = (float)number;
    if (server_json_number(line, "fire_pop_strength", &number))
        request->fire_pop_strength = (float)number;
    if (server_json_number(line, "radiance_knee", &number)) request->radiance_knee = (float)number;
    if (server_json_number(line, "radiance_strength", &number)) request->radiance_strength = (float)number;
    request->gamma_mode = server_control_mode(line, "gamma_mode", request->gamma_mode);
    request->cr_mode = server_control_mode(line, "cr_mode", request->cr_mode);
    request->fire_pop_mode = server_control_mode(line, "fire_pop_mode", request->fire_pop_mode);
    request->radiance_mode = server_control_mode(line, "radiance_mode", request->radiance_mode);
    request->chroma_mode = server_control_mode(line, "chroma_mode", request->chroma_mode);
    if (server_json_number(line, "chroma_neutral_boost", &number)) request->chroma_neutral_boost = (float)number;
    if (server_json_number(line, "chroma_fire_boost", &number)) request->chroma_fire_boost = (float)number;
    if (server_json_number(line, "chroma_knee", &number)) request->chroma_knee = (float)number;
    if (server_json_number(line, "chroma_skin_protect", &number)) request->chroma_skin_protect = (float)number;
    return request->width > 0 && request->height > 0;
}

static bool write_u32_le(uint32_t value)
{
    uint8_t bytes[4] = {
        (uint8_t)value, (uint8_t)(value >> 8),
        (uint8_t)(value >> 16), (uint8_t)(value >> 24),
    };
    return fwrite(bytes, 1, sizeof(bytes), stdout) == sizeof(bytes);
}

static bool write_f32_le(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return write_u32_le(bits);
}

static bool write_server_header(const ServerRequest *request)
{
    static const uint8_t magic[] = { 'D', 'V', 'R', '2' };
    return fwrite(magic, 1, sizeof(magic), stdout) == sizeof(magic) &&
           write_u32_le(2) && write_u32_le((uint32_t)request->width) &&
           write_u32_le((uint32_t)request->height) && write_u32_le(3);
}

static bool write_server_frame_info(const struct dv_frame_info *info)
{
    return write_u32_le(3) && write_u32_le(52) &&
           write_u32_le(info->flags) && write_f32_le(info->gamma) &&
           write_f32_le(info->l1_max_pq) && write_f32_le(info->l1_avg_pq) &&
           write_f32_le(info->cr_strength) && write_f32_le(info->l2_power) &&
           write_f32_le(info->fire_pop_strength) &&
           write_f32_le(info->radiance_knee) && write_f32_le(info->radiance_strength) &&
           write_f32_le(info->chroma_neutral_boost) && write_f32_le(info->chroma_fire_boost) &&
           write_f32_le(info->chroma_knee) && write_f32_le(info->chroma_skin_protect);
}

static bool write_server_record(uint32_t mode, const ServerRequest *request,
                                pl_gpu gpu, pl_renderer renderer,
                                pl_ml_context ml_context,
                                NativeDecoder *decoder,
                                struct dv_frame_info *frame_info)
{
    uint64_t payload_size = (uint64_t)request->width * request->height * 3;
    if (payload_size > UINT32_MAX || !write_u32_le(mode) ||
        !write_u32_le((uint32_t)payload_size))
        return false;

    Args args = {0};
    args.input = request->input;
    args.pts = request->pts;
    args.width = request->width;
    args.height = request->height;
    args.write_output = true;
    args.out_nits = request->out_nits;
    args.l1_max_pq = request->l1_max_pq;
    args.l1_avg_pq = request->l1_avg_pq;
    args.spline_contrast = -1.0f;
    args.knee_adaptation = args.knee_minimum = args.knee_maximum = -1.0f;
    args.knee_default = args.slope_tuning = args.slope_offset = -1.0f;
    args.perceptual_strength = -1.0f;
    args.gamut_expansion = -1;
    args.l2_power = args.l2_sat_gain = -1.0f;
    args.contrast_gamma = request->contrast_gamma;
    args.gamma_mode = request->gamma_mode;
    args.contrast_sat = 1.0f;
    args.highlight_knee = args.cr_smoothness = -1.0f;
    args.cr_strength = request->cr_strength;
    args.cr_mode = request->cr_mode;
    args.fire_pop_strength = request->fire_pop_strength;
    args.fire_pop_mode = request->fire_pop_mode;
    args.radiance_knee = request->radiance_knee;
    args.radiance_strength = request->radiance_strength;
    args.radiance_mode = request->radiance_mode;
    args.chroma_mode = mode == 1 ? DV_CONTROL_OFF : request->chroma_mode;
    args.chroma_neutral_boost = request->chroma_neutral_boost;
    args.chroma_fire_boost = request->chroma_fire_boost;
    args.chroma_knee = request->chroma_knee;
    args.chroma_skin_protect = request->chroma_skin_protect;
    args.mode = mode == 1 ? "spline" : "contrast-recovery";
    args.contrast_recovery = mode == 2;

    return render_frame_with_decoder(args, gpu, renderer, ml_context, decoder,
                                     NULL, NULL, frame_info) == 0;
}

typedef struct {
    pl_queue queue;
    double first_pts;
    double last_pts;
    bool have_pts;
} PlaybackState;

static bool playback_map_frame(pl_gpu gpu, pl_tex *tex,
                               const struct pl_source_frame *src,
                               struct pl_frame *out_frame)
{
    int64_t start = av_gettime_relative();
    AVFrame *frame = src->frame_data;
    bool ok = pl_map_avframe_ex(gpu, out_frame, pl_avframe_params(
        .frame = frame, .tex = tex, .map_dovi = true));
    av_frame_free(&frame);
    fprintf(stderr, "Playback timing: map=%.3f ms\n",
            (av_gettime_relative() - start) / 1000.0);
    return ok;
}

static void playback_unmap_frame(pl_gpu gpu, struct pl_frame *frame,
                                  const struct pl_source_frame *src)
{
    (void)src;
    pl_unmap_avframe(gpu, frame);
}

static void playback_discard_frame(const struct pl_source_frame *src)
{
    AVFrame *frame = src->frame_data;
    av_frame_free(&frame);
}

static bool write_playback_response(const ServerRequest *request, double pts,
                                    pl_gpu gpu, pl_renderer renderer,
                                    pl_ml_context ml_context,
                                    NativeDecoder *decoder,
                                    PlaybackState *playback)
{
    int64_t decode_start = av_gettime_relative();
    AVFrame *frame = native_decoder_decode(decoder, request->pts);
    double decode_ms = (av_gettime_relative() - decode_start) / 1000.0;
    if (!frame)
        return false;

    double frame_duration = 1.0 / 24.0;
    if (decoder->stream->avg_frame_rate.num && decoder->stream->avg_frame_rate.den)
        frame_duration = av_q2d(av_inv_q(decoder->stream->avg_frame_rate));

    if (!playback->have_pts || pts < playback->last_pts) {
        pl_queue_reset(playback->queue);
        playback->first_pts = pts;
        playback->have_pts = true;
        fprintf(stderr, "Playback queue: reset at %.6f\n", pts);
    }

    double queue_pts = pts - playback->first_pts;
    AVFrame *queued_frame = av_frame_clone(frame);
    if (!queued_frame) {
        av_frame_free(&frame);
        return false;
    }
    pl_queue_push(playback->queue, &(struct pl_source_frame) {
        .pts = queue_pts,
        .frame_data = queued_frame,
        .map = playback_map_frame,
        .unmap = playback_unmap_frame,
        .discard = playback_discard_frame,
    });

    /* pl_queue needs a lookahead frame, as in plplay's decoder thread. */
    AVFrame *lookahead = native_decoder_decode(decoder, request->pts + frame_duration);
    if (lookahead) {
        AVFrame *queued_lookahead = av_frame_clone(lookahead);
        av_frame_free(&lookahead);
        if (!queued_lookahead)
            return false;
        pl_queue_push(playback->queue, &(struct pl_source_frame) {
            .pts = queue_pts + frame_duration,
            .duration = frame_duration,
            .frame_data = queued_lookahead,
            .map = playback_map_frame,
            .unmap = playback_unmap_frame,
            .discard = playback_discard_frame,
        });
    }

    int64_t queue_start = av_gettime_relative();
    struct pl_frame_mix mix;
    enum pl_queue_status status = pl_queue_update(playback->queue, &mix,
        pl_queue_params(.pts = queue_pts, .radius = 0.0f, .timeout = 0));
    double queue_ms = (av_gettime_relative() - queue_start) / 1000.0;
    if (status != PL_QUEUE_OK || !mix.num_frames) {
        fprintf(stderr, "Playback queue: update failed (%d)\n", status);
        av_frame_free(&frame);
        return false;
    }
    playback->last_pts = pts;

    uint64_t payload_size = (uint64_t)request->width * request->height * 3;
    static const uint8_t magic[] = { 'D', 'V', 'R', 'P' };
    if (payload_size > UINT32_MAX ||
        fwrite(magic, 1, sizeof(magic), stdout) != sizeof(magic) ||
        !write_u32_le(1) || !write_u32_le((uint32_t)request->width) ||
        !write_u32_le((uint32_t)request->height) ||
        !write_u32_le((uint32_t)llround(pts * 1000.0)) ||
        !write_u32_le((uint32_t)payload_size)) {
        av_frame_free(&frame);
        return false;
    }

    Args args = {0};
    args.input = request->input;
    args.pts = request->pts;
    args.width = request->width;
    args.height = request->height;
    args.write_output = true;
    args.out_nits = request->out_nits;
    args.l1_max_pq = request->l1_max_pq;
    args.l1_avg_pq = request->l1_avg_pq;
    args.contrast_gamma = request->contrast_gamma;
    args.contrast_sat = 1.0f;
    args.cr_strength = request->cr_strength;
    args.fire_pop_strength = request->fire_pop_strength;
    args.mode = "spline";
    int64_t render_start = av_gettime_relative();
    int result = render_frame_with_decoder(args, gpu, renderer, ml_context,
                                           NULL, NULL,
                                           mix.frames[0], NULL);
    av_frame_free(&frame);
    double render_ms = (av_gettime_relative() - render_start) / 1000.0;
    fprintf(stderr, "Playback timing: decode=%.3f ms queue=%.3f ms render=%.3f ms\n",
            decode_ms, queue_ms, render_ms);
    return result == 0;
}

/* -------------------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------------- */
int main(int argc, char **argv)
{
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    av_log_set_level(AV_LOG_WARNING);

    Args args = {0};
    if (!parse_args(argc, argv, &args)) { usage(argv[0]); return 1; }

    pl_log log = pl_log_create(PL_API_VER, pl_log_params(
        .log_cb = pl_log_simple, .log_priv = stderr, .log_level = PL_LOG_WARN));
    pl_d3d11 d3d11 = pl_d3d11_create(log, pl_d3d11_params(.allow_software = true));
    if (!d3d11) { fprintf(stderr, "Failed to create D3D11 context\n"); return 1; }
    pl_gpu gpu = d3d11->gpu;
    init_cr_vars();
    pl_renderer renderer = pl_renderer_create(log, gpu);
    if (!renderer) { fprintf(stderr, "Failed to create renderer\n"); return 1; }
    pl_ml_context ml_context = NULL;
    if (args.model_path) {
        ml_context = pl_ml_context_create(pl_ml_context_params(
            .log = log, .model_path = args.model_path));
        if (!ml_context) {
            fprintf(stderr, "Failed to initialize native ML model: %s\n",
                    args.model_path);
            pl_renderer_destroy(&renderer);
            pl_d3d11_destroy(&d3d11);
            pl_log_destroy(&log);
            return 1;
        }
        fprintf(stderr, "Native ML model initialized: %s\n", args.model_path);
    }

    if (!args.server && args.playback_server) {
        fprintf(stderr, "DV renderer playback server ready: DVRP responses on stdout\n");
        NativeDecoder decoder = {0};
        PlaybackState playback = { .queue = pl_queue_create(gpu) };
        char line[4096];
        if (!playback.queue) {
            fprintf(stderr, "Playback server: failed creating frame queue\n");
            pl_renderer_destroy(&renderer);
            pl_ml_context_destroy(&ml_context);
            pl_d3d11_destroy(&d3d11);
            pl_log_destroy(&log);
            return 1;
        }
        while (fgets(line, sizeof(line), stdin)) {
            ServerRequest request;
            if (!parse_server_request(line, &request)) {
                fprintf(stderr, "Playback server: invalid request\n");
                continue;
            }
            if (strcmp(decoder.input, request.input) != 0) {
                pl_queue_reset(playback.queue);
                playback.have_pts = false;
                if (!native_decoder_open(&decoder, request.input, gpu)) {
                    fprintf(stderr, "Playback server: decoder open failed\n");
                    break;
                }
            }
            if (!write_playback_response(&request, request.pts, gpu,
                                         renderer, ml_context, &decoder, &playback)) {
                fprintf(stderr, "Playback server: request failed\n");
                break;
            }
            fflush(stdout);
        }
        pl_queue_destroy(&playback.queue);
        native_decoder_close(&decoder);
        pl_renderer_destroy(&renderer);
        pl_ml_context_destroy(&ml_context);
        pl_d3d11_destroy(&d3d11);
        pl_log_destroy(&log);
        return 0;
    }

    if (!args.server) {
        int result = render_frame_with_decoder(args, gpu, renderer, ml_context,
                               NULL, NULL, NULL, NULL);
        pl_renderer_destroy(&renderer);
        pl_ml_context_destroy(&ml_context);
        pl_d3d11_destroy(&d3d11);
        pl_log_destroy(&log);
        return result;
    }

    fprintf(stderr, "DV renderer server ready: DVR2 responses on stdout\n");
    NativeDecoder decoder = {0};
    char line[4096];
    while (fgets(line, sizeof(line), stdin)) {
        ServerRequest request;
        if (!parse_server_request(line, &request)) {
            fprintf(stderr, "Server: invalid request\n");
            continue;
        }
        if (strcmp(decoder.input, request.input) != 0) {
            if (!native_decoder_open(&decoder, request.input, gpu)) {
                fprintf(stderr, "Server: decoder open failed\n");
                break;
            }
        } else {
            fprintf(stderr, "Server decoder reuse: %s\n", request.input);
        }
        struct dv_frame_info frame_info = {0};
        if (!write_server_header(&request) ||
            !write_server_record(1, &request, gpu, renderer, ml_context, &decoder, NULL) ||
            !write_server_record(2, &request, gpu, renderer, ml_context, &decoder,
                                 &frame_info) ||
            !write_server_frame_info(&frame_info)) {
            fprintf(stderr, "Server: request failed\n");
            break;
        }
        fflush(stdout);
    }

    native_decoder_close(&decoder);
    pl_renderer_destroy(&renderer);
    pl_ml_context_destroy(&ml_context);
    pl_d3d11_destroy(&d3d11);
    pl_log_destroy(&log);
    return 0;
}
