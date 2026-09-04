#include <math.h>
#include <stdio.h>

#include <libplacebo/dispatch.h>
#include <libplacebo/ml_render.h>
#include <libplacebo/tone_mapping.h>
#include <libplacebo/shaders/colorspace.h>

// Reference transfer used for all PL_HOOK_OUTPUT Zion Core grades.
// The L2 gamma, radiance and chroma-tuner shaders were tuned assuming they
// operate on a gamma-2.2 encoded 0..1 signal. Wrapping the hooks with a
// target-transfer <-> gamma-2.2 round-trip makes the grade independent of
// --target-trc and fixes the compounding that blows out highlights on SDR.
#define ML_GRADE_REF_TRANSFER PL_COLOR_TRC_GAMMA22

void ml_grade_to_ref(pl_shader sh,
                     const struct pl_color_space *target_csp)
{
    if (target_csp->transfer == ML_GRADE_REF_TRANSFER)
        return;
    struct pl_color_space ref_csp = *target_csp;
    ref_csp.transfer = ML_GRADE_REF_TRANSFER;
    pl_shader_linearize(sh, target_csp);
    pl_shader_delinearize(sh, &ref_csp);
}

void ml_grade_from_ref(pl_shader sh,
                       const struct pl_color_space *target_csp)
{
    if (target_csp->transfer == ML_GRADE_REF_TRANSFER)
        return;
    struct pl_color_space ref_csp = *target_csp;
    ref_csp.transfer = ML_GRADE_REF_TRANSFER;
    pl_shader_linearize(sh, &ref_csp);
    pl_shader_delinearize(sh, target_csp);
}

// ── SDR → true-PQ helpers (ST.2084 constants; not exported by public headers) ──
#define SDR_VIRTUAL_PEAK_PQ 0.7518f   // 1000-nit virtual P5 ceiling
#define SDR_FEATURE0_FLOOR  0.62f     // feature-87 compression-floor on SDR

static inline float sdr_pq_to_nits(float pq)
{
    if (pq <= 0.0f) return 0.0f;
    float p = powf(pq, 1.0f / 78.84375f);   // m2 = 2523/32
    float num = fmaxf(p - 0.8359375f, 0.0f); // c1 = 3424/4096
    float den = fmaxf(18.8515625f - 18.6875f * p, 1e-6f); // c2, c3
    return powf(num / den, 1.0f / 0.1593017578125f) * 10000.0f; // m1
}

static inline float sdr_nits_to_pq(float nits)
{
    if (nits <= 0.0f) return 0.0f;
    float n = fminf(fmaxf(nits / 10000.0f, 1e-6f), 1.0f);
    float m = powf(n, 0.1593017578125f);
    return powf((0.8359375f + 18.8515625f * m) / (1.0f + 18.6875f * m),
                78.84375f);
}

static struct pl_hook_res l2_hook(void *priv, const struct pl_hook_params *params)
{
    struct pl_ml_render_result *result = priv;
    pl_shader sh = params->sh;
    const struct pl_color_space target_csp = params->color;
    static const char body[] =
        "float y = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));\n"
        "float cb = (color.b - y) * (l2_saturation / 2048.0);\n"
        "float cr = (color.r - y) * (l2_saturation / 2048.0);\n"
        "float y_pre = y;\n"
        "float l2_gamma = 2048.0 / l2_power;\n"
        "if (l2_gamma != 1.0 && y > 0.001 && y < 0.999) {\n"
        "    float hi = smoothstep(l2_guard, 0.98, y);\n"
        "    if (l2_gamma > 1.0) {\n"
        "        float fade = smoothstep(0.08, 0.45, y);\n"
        "        float eff = fade * (1.0 - hi);\n"
        "        y = mix(y, pow(y, 1.0 / l2_gamma), eff);\n"
        "    } else {\n"
        "        float fade2 = smoothstep(0.08, 0.45, y);\n"
        "        float g = mix(1.0, l2_gamma, fade2 * (1.0 - hi));\n"
        "        y = pow(y, g);\n"
        "    }\n"
        "}\n"
        "if (l2_dehaze > 0.01 && y > 0.001 && y < l2_dehaze_knee) {\n"
        "    float t = y / l2_dehaze_knee;\n"
        "    float fade = 1.0 - t * t;\n"
        "    y = l2_dehaze_knee * pow(t, 1.0 + l2_dehaze * 2.0 * fade);\n"
        "}\n"
        "float hunt = 1.0 + l2_hunt * pow(l2_l1max, 2.0);\n"
        "if (y > 0.001 && y_pre > 0.001) {\n"
        "    float gamma_comp = pow(y_pre / y, 0.15);\n"
        "    cr *= hunt * gamma_comp;\n"
        "    cb *= hunt * gamma_comp;\n"
        "}\n"
        "float desat = smoothstep(l2_guard, 0.95, y);\n"
        "cr *= (1.0 - desat * l2_hi_desat);\n"
        "cb *= (1.0 - desat * l2_hi_desat);\n"
        "if (abs(l2_hue_deg) > 0.1) {\n"
        "    float theta = l2_hue_deg * 0.017453293;\n"
        "    float hc = cos(theta), hs = sin(theta);\n"
        "    float cr2 = cr * hc - cb * hs;\n"
        "    float cb2 = cr * hs + cb * hc;\n"
        "    cr = cr2; cb = cb2;\n"
        "}\n"
        "float r_out = y + cr;\n"
        "float g_out = y - 0.2126 / 0.7152 * cr - 0.0722 / 0.7152 * cb;\n"
        "float b_out = y + cb;\n"
        "float mx = max(max(r_out, g_out), b_out);\n"
        "float mn = min(min(r_out, g_out), b_out);\n"
        "if (mx > 1.0 || mn < 0.0) {\n"
        "    float s = 1.0;\n"
        "    if (mx > 1.0 && mx > y + 0.001) s = min(s, (1.0 - y) / (mx - y));\n"
        "    if (mn < 0.0 && mn < y - 0.001) s = min(s, y / (y - mn));\n"
        "    cr *= max(s, 0.0); cb *= max(s, 0.0);\n"
        "}\n"
        "color.r = clamp(y + cr, 0.0, 1.0);\n"
        "color.g = clamp(y - 0.2126 / 0.7152 * cr - 0.0722 / 0.7152 * cb, 0.0, 1.0);\n"
        "color.b = clamp(y + cb, 0.0, 1.0);\n";
    struct pl_shader_var vars[] = {
        { .var = pl_var_float("l2_power"), .data = &result->l2_power, .dynamic = true },
        { .var = pl_var_float("l2_saturation"), .data = &result->l2_saturation, .dynamic = true },
        { .var = pl_var_float("l2_guard"), .data = &result->l2_highlight_guard, .dynamic = true },
        { .var = pl_var_float("l2_hue_deg"), .data = &result->hue_correction, .dynamic = true },
        { .var = pl_var_float("l2_dehaze"), .data = &result->dehaze_strength, .dynamic = true },
        { .var = pl_var_float("l2_dehaze_knee"), .data = &result->dehaze_knee, .dynamic = true },
        { .var = pl_var_float("l2_hunt"), .data = &result->hunt_strength, .dynamic = true },
        { .var = pl_var_float("l2_l1max"), .data = &result->l1_max_pq, .dynamic = true },
        { .var = pl_var_float("l2_hi_desat"), .data = &result->highlight_desat, .dynamic = true },
    };
    ml_grade_to_ref(sh, &target_csp);
    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "GPU L2 gamma and saturation trim", .body = body,
        .input = PL_SHADER_SIG_COLOR, .output = PL_SHADER_SIG_COLOR,
        .variables = vars, .num_variables = 9,
        .output_w = pl_rect_w(params->dst_rect), .output_h = pl_rect_h(params->dst_rect),
    })) return (struct pl_hook_res) { .failed = true };
    ml_grade_from_ref(sh, &target_csp);
    return (struct pl_hook_res) { .output = PL_HOOK_SIG_COLOR, .sh = sh,
        .repr = params->repr, .color = params->color, .components = params->components,
        .rect = params->rect };
}

static struct pl_hook_res fire_hook(void *priv, const struct pl_hook_params *params)
{
    struct pl_ml_render_result *result = priv;
    pl_shader sh = params->sh;
    const struct pl_color_space target_csp = params->color;
    static const char body[] =
        "float y = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));\n"
        "float t = clamp((y - 0.45) / 0.40, 0.0, 1.0);\n"
        "float s = t * t * (3.0 - 2.0 * t);\n"
        "float envelope = sin(3.141592653589793 * s); envelope *= envelope;\n"
        "float cr = color.r - y; float cb = color.b - y;\n"
        "if (y > 0.45 && cr > 0.08 && cb < -0.01) {\n"
        " cr *= 1.0 + 0.12 * envelope * fire_strength;\n"
        " cb *= 1.0 + 0.08 * envelope * fire_strength;\n"
        " color.r = clamp(y + cr, 0.0, 1.0); color.b = clamp(y + cb, 0.0, 1.0);\n"
        " color.g = clamp(y - 0.2126 / 0.7152 * cr - 0.0722 / 0.7152 * cb, 0.0, 1.0);\n"
        "}\n";
    struct pl_shader_var var = { .var = pl_var_float("fire_strength"),
        .data = &result->fire_pop_strength, .dynamic = true };
    ml_grade_to_ref(sh, &target_csp);
    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "GPU fire-pop output hook", .body = body,
        .input = PL_SHADER_SIG_COLOR, .output = PL_SHADER_SIG_COLOR,
        .variables = &var, .num_variables = 1,
        .output_w = pl_rect_w(params->dst_rect), .output_h = pl_rect_h(params->dst_rect),
    })) return (struct pl_hook_res) { .failed = true };
    ml_grade_from_ref(sh, &target_csp);
    return (struct pl_hook_res) { .output = PL_HOOK_SIG_COLOR, .sh = sh,
        .repr = params->repr, .color = params->color, .components = params->components,
        .rect = params->rect };
}

static struct pl_hook_res chroma_tuner_hook(void *priv, const struct pl_hook_params *params)
{
    struct pl_ml_render_result *result = priv;
    pl_shader sh = params->sh;
    // Gamma-2.2 chroma tuner.
    //
    // ml_grade_to_ref already converts the signal to gamma-2.2 encoded [0,1]
    // before this body runs. The cr/cb decomposition and skin ellipse operate
    // directly on these gamma-encoded values, matching the SDR training domain
    // (WIDER FACE, sRGB ≈ gamma 2.2).  No extra pow() is needed.
    static const char body_a[] =
        // 1. Decompose gamma-2.2 signal into luma + chroma residuals
        "float y = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));\n"
        "float r_res = color.r - y;\n"
        "float b_res = color.b - y;\n"
        "float cb = b_res / 1.8556;\n"
        "float cr = r_res / 1.5748;\n"
        // 2. Skin ellipse (center/radii from WIDER FACE training stats in gamma-2.2)
        "float dr = (cr - 0.08) / 0.07;\n"
        "float db = (cb + 0.05) / 0.05;\n"
        "if (y > u_chroma_knee) {\n"
        "    float hd = clamp((y - u_chroma_knee) / (1.0 - u_chroma_knee), 0.0, 1.0);\n"
        "    float ex = 1.0 + hd * 0.50;\n"
        "    dr /= ex; db /= ex;\n"
        "}\n"
        "float skin_ellipse = dr * dr + db * db;\n"
        "float skin_weight = 1.0 / (1.0 + skin_ellipse * 12.0);\n"
        // 3. Highlight skin taper (gamma-2.2 space, y=0.60 ≈ 33% linear)
        "if (y > 0.60) {\n"
        "    float dt = clamp((y - 0.60) / 0.40, 0.0, 1.0);\n"
        "    float pe = dt * dt * (3.0 - 2.0 * dt);\n"
        "    skin_weight = mix(skin_weight, 1.0, pe);\n"
        "}\n"
        // 4. Compute adaptive chroma boost
        "float luma_boost = u_chroma_neutral_boost;\n"
        "float denom = 1.0 - u_chroma_knee;\n"
        "if (y > u_chroma_knee && denom > 0.001) {\n"
        "    float t = clamp((y - u_chroma_knee) / denom, 0.0, 1.0);\n"
        "    luma_boost = mix(u_chroma_neutral_boost, u_chroma_fire_boost, pow(t, 1.5));\n"
        "}\n"
        "float chroma_scalar = mix(luma_boost, 1.0, clamp(skin_weight * u_chroma_skin_protect, 0.0, 1.0));\n"
        "chroma_scalar = clamp(chroma_scalar, 1.0, 1.50);\n"
        // 5. Safe taper: prevent green going negative on saturated fire/highlights
        "float g_denom = (0.2126 / 0.7152) * r_res + (0.0722 / 0.7152) * b_res;\n"
        "if (g_denom > 0.0) {\n"
        "    float max_safe = y / g_denom;\n"
        "    if (max_safe < chroma_scalar) chroma_scalar = max(1.0, max_safe);\n"
        "}\n"
        // 6. Apply scalar to residuals, reconstruct gamma-2.2 signal
        "r_res *= chroma_scalar; b_res *= chroma_scalar;\n"
        "color.r = clamp(y + r_res, 0.0, 1.0);\n"
        "color.g = clamp(y - (0.2126 / 0.7152) * r_res - (0.0722 / 0.7152) * b_res, 0.0, 1.0);\n"
        "color.b = clamp(y + b_res, 0.0, 1.0);\n";
    // Body B — trained P(skin) LUT lookup instead of the hand-tuned ellipse.
    // No adaptive ellipse and NO highlight taper: the trained distribution
    // already spans bright scenes, so the mask survives them (the OLD taper
    // disabled protection exactly where the fire boost is largest).
    // ml_grade_to_ref already provides gamma-2.2 encoded values matching the
    // WIDER FACE training domain — no extra pow() needed.
    static const char body_b[] =
        // 1. Decompose gamma-2.2 signal into luma + chroma residuals
        "float y = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));\n"
        "float r_res = color.r - y;\n"
        "float b_res = color.b - y;\n"
        "float cb = b_res / 1.8556;\n"
        "float cr = r_res / 1.5748;\n"
        // 2. Trained P(skin) LUT lookup (cr→u, cb→v), bilinear + clamp.
        "vec2 uv = vec2((cr - u_skin_cr0) / (u_skin_cr1 - u_skin_cr0),\n"
        "               (cb - u_skin_cb0) / (u_skin_cb1 - u_skin_cb0));\n"
        "float skin_weight = textureLod(skin_lut, clamp(uv, 0.0, 1.0), 0.0).r;\n"
        // 3. Compute adaptive chroma boost
        "float luma_boost = u_chroma_neutral_boost;\n"
        "float denom = 1.0 - u_chroma_knee;\n"
        "if (y > u_chroma_knee && denom > 0.001) {\n"
        "    float t = clamp((y - u_chroma_knee) / denom, 0.0, 1.0);\n"
        "    luma_boost = mix(u_chroma_neutral_boost, u_chroma_fire_boost, pow(t, 1.5));\n"
        "}\n"
        "float chroma_scalar = mix(luma_boost, 1.0, clamp(skin_weight * u_chroma_skin_protect, 0.0, 1.0));\n"
        "chroma_scalar = clamp(chroma_scalar, 1.0, 1.50);\n"
        // 4. Safe taper: prevent green going negative on saturated fire/highlights
        "float g_denom = (0.2126 / 0.7152) * r_res + (0.0722 / 0.7152) * b_res;\n"
        "if (g_denom > 0.0) {\n"
        "    float max_safe = y / g_denom;\n"
        "    if (max_safe < chroma_scalar) chroma_scalar = max(1.0, max_safe);\n"
        "}\n"
        // 4b. Apply scalar to residuals, reconstruct gamma-2.2 signal
        "r_res *= chroma_scalar; b_res *= chroma_scalar;\n"
        "color.r = clamp(y + r_res, 0.0, 1.0);\n"
        "color.g = clamp(y - (0.2126 / 0.7152) * r_res - (0.0722 / 0.7152) * b_res, 0.0, 1.0);\n"
        "color.b = clamp(y + b_res, 0.0, 1.0);\n";
    struct pl_shader_var vars[] = {
        { .var = pl_var_float("u_chroma_neutral_boost"), .data = &result->chroma_neutral_boost, .dynamic = true },
        { .var = pl_var_float("u_chroma_fire_boost"),    .data = &result->chroma_fire_boost,    .dynamic = true },
        { .var = pl_var_float("u_chroma_knee"),          .data = &result->chroma_knee,          .dynamic = true },
        { .var = pl_var_float("u_chroma_skin_protect"),  .data = &result->chroma_skin_protect,  .dynamic = true },
        // body-B LUT grid bounds (registered only when lut_ok)
        { .var = pl_var_float("u_skin_cr0"), .data = &result->skin_lut_cr0, .dynamic = true },
        { .var = pl_var_float("u_skin_cr1"), .data = &result->skin_lut_cr1, .dynamic = true },
        { .var = pl_var_float("u_skin_cb0"), .data = &result->skin_lut_cb0, .dynamic = true },
        { .var = pl_var_float("u_skin_cb1"), .data = &result->skin_lut_cb1, .dynamic = true },
    };
    bool lut_ok = result->skin_lut != NULL && result->skin_lut_gpu == params->gpu;
    const char *body = lut_ok ? body_b : body_a;
    struct pl_shader_desc desc = lut_ok ? (struct pl_shader_desc) {
        .desc = { .name = "skin_lut", .type = PL_DESC_SAMPLED_TEX },
        .binding = {
            .object = result->skin_lut,
            .sample_mode  = PL_TEX_SAMPLE_LINEAR,   // bilinear P(skin)
            .address_mode = PL_TEX_ADDRESS_CLAMP,
        },
    } : (struct pl_shader_desc) {0};
    const struct pl_color_space target_csp = params->color;
    ml_grade_to_ref(sh, &target_csp);
    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "Adaptive Chroma Vector Tuner", .body = body,
        .input = PL_SHADER_SIG_COLOR, .output = PL_SHADER_SIG_COLOR,
        .variables = vars, .num_variables = lut_ok ? 8 : 4,
        .descriptors = lut_ok ? &desc : NULL,
        .num_descriptors = lut_ok ? 1 : 0,
    })) return (struct pl_hook_res) { .failed = true };
    ml_grade_from_ref(sh, &target_csp);
    return (struct pl_hook_res) { .output = PL_HOOK_SIG_COLOR, .sh = sh,
        .repr = params->repr, .color = params->color, .components = params->components,
        .rect = params->rect };
}

// PQ pre-compressor — two-stage tone mapping: soft-clips the PQ signal to a
// virtual ceiling (e.g. 203 nits ≈ PQ 0.5810) using Reinhard on the PQ
// PQ pre-compressor: soft-clips highlight PQ values above the virtual
// ceiling so inter-pixel gradients (clouds, sand, specular) survive the
// subsequent tone map. Only modifies pixels — leaves max_luma metadata
// untouched so the tone mapper keeps its normal aggressive curve.
static struct pl_hook_res pq_precompress_hook(void *priv,
                                              const struct pl_hook_params *hp)
{
    struct pl_ml_render_result *result = priv;
    if (result->dc_ceiling_pq <= 0.001f)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };
    if (hp->color.transfer != PL_COLOR_TRC_PQ)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    pl_shader sh = hp->sh;

    static const char body[] =
        "float y_pq = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));\n"
        "float knee_pq = dc_ceil * 0.85;\n"
        "if (y_pq > knee_pq && y_pq > 0.001) {\n"
        "    float over = y_pq - knee_pq;\n"
        "    float room = dc_ceil - knee_pq;\n"
        "    float y_new = knee_pq + room * over / (over + room);\n"
        "    color.rgb *= y_new / y_pq;\n"
        "}\n";

    struct pl_shader_var vars[] = {
        { .var = pl_var_float("dc_ceil"), .data = &result->dc_ceiling_pq, .dynamic = true },
    };

    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "PQ pre-compress (two-stage TM)",
        .body        = body,
        .input       = PL_SHADER_SIG_COLOR,
        .output      = PL_SHADER_SIG_COLOR,
        .variables   = vars, .num_variables = 1,
    })) {
        return (struct pl_hook_res) { .failed = true };
    }

    return (struct pl_hook_res) {
        .output = PL_HOOK_SIG_COLOR, .sh = sh,
        .repr = hp->repr, .color = hp->color, .components = hp->components,
        .rect = hp->rect,
    };
}

// Highlight headroom — PL_HOOK_OUTPUT, runs FIRST before all other OUTPUT
// hooks. Applies a soft shoulder compression so downstream ML hooks
// (fire/pop, L2, CR, radiance) have room to boost without clipping.
// Below the knee (0.65 SDR luma) the signal passes untouched; above, it's
// smoothly compressed to leave 'headroom' fraction of the [0,1] range.
static struct pl_hook_res headroom_hook(void *priv,
                                        const struct pl_hook_params *hp)
{
    struct pl_ml_render_result *result = priv;
    if (result->headroom <= 0.001f)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    pl_shader sh = hp->sh;

    struct pl_shader_var vars[] = {
        { .var = pl_var_float("hr"), .data = &result->headroom, .dynamic = true },
    };

    static const char body[] =
        "float y = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));\n"
        "float knee = 0.65;\n"
        "if (y > knee && y > 0.001) {\n"
        "    float blend = smoothstep(knee, 1.0, y);\n"
        "    float new_y = y - hr * (y - knee) * blend;\n"
        "    color.rgb *= new_y / y;\n"
        "}\n";

    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "Highlight headroom",
        .body        = body,
        .input       = PL_SHADER_SIG_COLOR,
        .output      = PL_SHADER_SIG_COLOR,
        .variables   = vars, .num_variables = 1,
        .output_w    = pl_rect_w(hp->dst_rect),
        .output_h    = pl_rect_h(hp->dst_rect),
    })) {
        return (struct pl_hook_res) { .failed = true };
    }
    return (struct pl_hook_res) {
        .output = PL_HOOK_SIG_COLOR, .sh = sh,
        .repr = hp->repr, .color = hp->color, .components = hp->components,
        .rect = hp->rect,
    };
}

// Display compress — PL_HOOK_OUTPUT, runs LAST after all other OUTPUT hooks.
// Stage 2 of two-stage tone mapping: the tone mapper mapped to virtual_nits
// (e.g. 203); this hook soft-compresses the SDR signal down to real display
// peak (e.g. 51 nits). Uses a normalized Reinhard so highlight gradients
// from the gentle stage-1 mapping survive the final compress.
static struct pl_hook_res display_compress_hook(void *priv,
                                                const struct pl_hook_params *hp)
{
    struct pl_ml_render_result *result = priv;
    if (result->dc_compress_gain <= 1.001f)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    pl_shader sh = hp->sh;

    struct pl_shader_var vars[] = {
        { .var = pl_var_float("gain"), .data = &result->dc_compress_gain, .dynamic = true },
    };

    static const char hdr[] =
        "float hable(float x) {\n"
        "    return ((x * (0.15 * x + 0.05) + 0.004) /\n"
        "            (x * (0.15 * x + 0.50) + 0.06)) - 0.0667;\n"
        "}\n";

    static const char body[] =
        "float y = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));\n"
        "if (y > 0.001) {\n"
        "    float y_new = hable(y * gain) / hable(gain);\n"
        "    color.rgb *= y_new / y;\n"
        "}\n";

    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "Display compress (two-stage TM)",
        .header      = hdr,
        .body        = body,
        .input       = PL_SHADER_SIG_COLOR,
        .output      = PL_SHADER_SIG_COLOR,
        .variables   = vars, .num_variables = 1,
        .output_w    = pl_rect_w(hp->dst_rect),
        .output_h    = pl_rect_h(hp->dst_rect),
    })) {
        return (struct pl_hook_res) { .failed = true };
    }
    return (struct pl_hook_res) {
        .output = PL_HOOK_SIG_COLOR, .sh = sh,
        .repr = hp->repr, .color = hp->color, .components = hp->components,
        .rect = hp->rect,
    };
}

// Shadow bilateral — PL_HOOK_NATIVE (PQ signal, pre-linearization).
// 5×5 bilateral filter active where y_pq < shadow_knee (≈ 11.7 nits).
// Sigma-spatial: fixed 1.5 px.  Sigma-range: from shadow_strength.
// Uses PL_HOOK_SIG_TEX I/O: dispatches its own full-image render pass so it
// can sample neighboring pixels (not possible with PL_HOOK_SIG_COLOR).
//
// The learned toe un-crushes dark gradients via a multiplicative gain
//   out = rgb * (1 + str * (1 - y_pq/knee)^2)   (y_pq < knee)
// anchored at black (out(0)=0 → no fog/veil), slope at black = 1+str > 1
// (near-black detail contrast *expands*), identity with slope 1 at the knee
// (C1-continuous, monotonic).  Runs at PL_HOOK_OUTPUT (display-referred, post-
// tone-mapping) so the gain directly maps to visible brightness change.
static struct pl_hook_res shadow_bilateral_hook(void *priv,
                                                const struct pl_hook_params *params)
{
    struct pl_ml_render_result *result = priv;
    if (result->shadow_strength <= 0.001f && result->shadow_toe <= 0.001f)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    pl_tex src = params->tex;
    int w = src->params.w, h = src->params.h;
    pl_tex dst = params->get_tex(params->priv, w, h);
    if (!dst)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    pl_shader sh = pl_dispatch_begin(params->dispatch);

    // OUTPUT stage: always BT.709 (display-referred after color management)
    float luma[3] = { 0.2126f, 0.7152f, 0.0722f };

    struct pl_shader_var vars[] = {
        { .var = pl_var_float("shadow_toe"),  .data = &result->shadow_toe,      .dynamic = true },
        { .var = pl_var_float("shadow_str"),  .data = &result->shadow_strength, .dynamic = true },
        { .var = pl_var_float("shadow_knee"), .data = &result->shadow_knee,     .dynamic = true },
        { .var = pl_var_vec3("luma_w"),       .data = luma },
    };
    struct pl_shader_desc desc = {
        .desc = { .name = "shadow_src", .type = PL_DESC_SAMPLED_TEX },
        .binding = { .object = src,
                     .address_mode = PL_TEX_ADDRESS_CLAMP,
                     .sample_mode  = PL_TEX_SAMPLE_NEAREST },
    };
    // Shadow recovery: two-stage pass.
    // Stage 1 — black-anchored toe-lift: out = y + toe * (y/knee) * (1-y/knee)^2
    //   y=0 → 0 (black preserved), peak lift at y=knee/3, identity at knee.
    //   Ratio bounded at 1+toe/knee, no blowup on near-black pixels.
    // Stage 2 — log-space bilateral detail injection (existing):
    //   9×9 bilateral, stride=8 — restores local contrast in shadow gradients.
    static const char shadow_body[] =
        "vec2 sz = vec2(textureSize(shadow_src, 0));\n"
        "vec2 uv = gl_FragCoord.xy / sz;\n"
        "vec2 pt = 1.0 / sz;\n"
        "vec4 csrc = textureLod(shadow_src, uv, 0.0);\n"
        "vec3 center = csrc.rgb;\n"
        "float y_c = dot(center, luma_w);\n"
        "float smask = 1.0 - smoothstep(shadow_knee * 0.5, shadow_knee, y_c);\n"
        "smask *= smoothstep(0.0, 33.0 / sz.y, min(uv.y, 1.0 - uv.y))\n"
        "       * smoothstep(0.0, 33.0 / sz.x, min(uv.x, 1.0 - uv.x));\n"
        "vec3 out_rgb = center;\n"
        "if (shadow_toe > 0.001 && y_c < shadow_knee && y_c > 0.001) {\n"
        "    float t = 1.0 - y_c / shadow_knee;\n"
        "    float ramp = y_c / shadow_knee;\n"
        "    float y_lifted = y_c + shadow_toe * ramp * t * t;\n"
        "    float ratio_toe = y_lifted / y_c;\n"
        "    out_rgb = clamp(center * ratio_toe, 0.0, 1.0);\n"
        "    y_c = y_lifted;\n"
        "}\n"
        "if (smask > 0.001 && shadow_str > 0.001) {\n"
        "    float eps = 1e-4;\n"
        "    float log_yc = log(y_c + eps);\n"
        "    float log_acc = 0.0; float tw = 0.0;\n"
        "    float sr = 0.40;\n"
        "    for (int dy = -4; dy <= 4; dy++) {\n"
        "        for (int dx = -4; dx <= 4; dx++) {\n"
        "            float sy = dot(textureLod(shadow_src, uv + vec2(dx, dy) * pt * 8.0, 0.0).rgb, luma_w);\n"
        "            if (sy > 0.002) {\n"
        "                float log_sy = log(sy + eps);\n"
        "                float d2 = float(dx*dx + dy*dy);\n"
        "                float ld = log_yc - log_sy;\n"
        "                float w = exp(-d2 * 0.0556) * exp(-ld*ld / (2.0*sr*sr));\n"
        "                log_acc += w * log_sy; tw += w;\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "    float log_blur = log_acc / max(tw, 0.001);\n"
        "    float detail = log_yc - log_blur;\n"
        "    float log_new = log_yc + shadow_str * detail;\n"
        "    float y_new = clamp(exp(log_new) - eps, 0.0, 1.0);\n"
        "    float ratio = (y_c > eps) ? y_new / y_c : 1.0;\n"
        "    out_rgb = mix(out_rgb, clamp(out_rgb * ratio, 0.0, 1.0), smask);\n"
        "}\n"
        "color = vec4(out_rgb, csrc.a);\n";

    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "Shadow toe-lift + contrast recovery",
        .body        = shadow_body,
        .input       = PL_SHADER_SIG_NONE,
        .output      = PL_SHADER_SIG_COLOR,
        .variables   = vars, .num_variables   = 4,
        .descriptors = &desc, .num_descriptors = 1,
        .output_w    = w,     .output_h        = h,
    })) {
        pl_dispatch_abort(params->dispatch, &sh);
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };
    }
    if (!pl_dispatch_finish(params->dispatch,
            pl_dispatch_params(.shader = &sh, .target = dst)))
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    return (struct pl_hook_res) {
        .output     = PL_HOOK_SIG_TEX,
        .tex        = dst,
        .repr       = params->repr,
        .color      = params->color,
        .components = params->components,
        .rect       = params->rect,
    };
}

// SDR→P5 virtual-master bridge — PL_HOOK_RGB_INPUT, runs BEFORE the shadow
// filter so every downstream stage (shadow mask, peak/feature accumulation,
// tone mapping, ML grade) sees a virtual DV-style HDR master instead of raw
// SDR. Decodes the SDR (gamma/bt.1886) signal to relative linear, scales to
// the virtual P5 ceiling (sdr_virtual_nits), re-encodes as PQ and re-labels
// the frame PQ/sig_peak=virtual. (ml_render.h docs.)
static struct pl_hook_res sdr_p5_hook(void *priv, const struct pl_hook_params *params)
{
    struct pl_ml_render_result *result = priv;
    if (pl_color_space_is_hdr(params->orig_color))
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    pl_tex src = params->tex;
    int w = src->params.w, h = src->params.h;
    pl_tex dst = params->get_tex(params->priv, w, h);
    if (!dst)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    float virtual_nits = result->sdr_virtual_nits > 0.0f
                       ? result->sdr_virtual_nits : 1000.0f;
    float strength = fmaxf(0.0f, fminf(1.0f, result->sdr_strength));

    const char *decode;
    switch (params->color.transfer) {
    case PL_COLOR_TRC_LINEAR:
        decode = "vec3 rel = rgb;";
        break;
    case PL_COLOR_TRC_SRGB:
        decode = "vec3 sc = max(rgb, vec3(0.0));\n"
                 "vec3 lo = sc / 12.92;\n"
                 "vec3 hi = pow((sc + 0.055) / 1.055, vec3(2.4));\n"
                 "vec3 rel = mix(lo, hi, step(vec3(0.04045), sc));";
        break;
    case PL_COLOR_TRC_BT_1886:
    case PL_COLOR_TRC_GAMMA24:
        decode = "vec3 rel = pow(max(rgb, vec3(0.0)), vec3(2.4));";
        break;
    case PL_COLOR_TRC_GAMMA18:
    case PL_COLOR_TRC_PRO_PHOTO:
        decode = "vec3 rel = pow(max(rgb, vec3(0.0)), vec3(1.8));";
        break;
    case PL_COLOR_TRC_GAMMA20:
        decode = "vec3 rel = pow(max(rgb, vec3(0.0)), vec3(2.0));";
        break;
    case PL_COLOR_TRC_GAMMA22:
        decode = "vec3 rel = pow(max(rgb, vec3(0.0)), vec3(2.2));";
        break;
    case PL_COLOR_TRC_GAMMA26:
    case PL_COLOR_TRC_ST428:
        decode = "vec3 rel = pow(max(rgb, vec3(0.0)), vec3(2.6));";
        break;
    case PL_COLOR_TRC_GAMMA28:
        decode = "vec3 rel = pow(max(rgb, vec3(0.0)), vec3(2.8));";
        break;
    default:
        decode = "vec3 rel = pow(max(rgb, vec3(0.0)), vec3(2.2));";
        break;
    }

    char body[1024];
    snprintf(body, sizeof(body),
        "vec2 psz = vec2(textureSize(sdr_ml_src, 0));\n"
        "vec2 puv = gl_FragCoord.xy / psz;\n"
        "vec4 pc  = textureLod(sdr_ml_src, puv, 0.0);\n"
        "vec3 rgb = pc.rgb;\n"
        "%s\n"
        "vec3 plasm = (sdr_ml_str > 0.001)\n"
        "    ? pow(max(rel, vec3(0.0)), vec3(1.0 / (1.0 + sdr_ml_str)))\n"
        "    : rel;\n"
        "vec3 pvn = plasm * sdr_ml_vps;\n"
        "vec3 pn  = clamp(pvn / 10000.0, 0.0, 1.0);\n"
        "vec3 pm  = pow(pn, vec3(0.1593017578125));\n"
        "vec3 ppq = pow((vec3(0.8359375) + vec3(18.8515625) * pm)\n"
        "             / (vec3(1.0) + vec3(18.6875) * pm), vec3(78.84375));\n"
        "color = vec4(ppq, pc.a);\n",
        decode);

    struct pl_shader_var vars[] = {
        { .var = pl_var_float("sdr_ml_vps"), .data = &virtual_nits, .dynamic = true },
        { .var = pl_var_float("sdr_ml_str"), .data = &strength,     .dynamic = true },
    };
    struct pl_shader_desc desc = {
        .desc = { .name = "sdr_ml_src", .type = PL_DESC_SAMPLED_TEX },
        .binding = { .object = src, .address_mode = PL_TEX_ADDRESS_CLAMP,
                     .sample_mode = PL_TEX_SAMPLE_NEAREST },
    };

    pl_shader sh = pl_dispatch_begin(params->dispatch);
    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "SDR→P5 virtual master",
        .body        = body,
        .input       = PL_SHADER_SIG_NONE,
        .output      = PL_SHADER_SIG_COLOR,
        .variables   = vars, .num_variables   = 2,
        .descriptors = &desc, .num_descriptors = 1,
        .output_w    = w,     .output_h        = h,
    })) {
        pl_dispatch_abort(params->dispatch, &sh);
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };
    }
    if (!pl_dispatch_finish(params->dispatch,
            pl_dispatch_params(.shader = &sh, .target = dst)))
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    struct pl_color_space out_color = params->color;
    out_color.transfer = PL_COLOR_TRC_PQ;
    out_color.hdr.max_luma = virtual_nits;   // > SDR white → treated as HDR

    return (struct pl_hook_res) {
        .output = PL_HOOK_SIG_TEX,
        .tex    = dst,
        .repr   = params->repr,
        .color  = out_color,
        .components = params->components,
        .rect   = params->rect,
    };
}

// Highlight bilateral — PL_HOOK_OUTPUT (post-curve, display output space).
// 5×5 bilateral active where y > highlight_knee. Tighter sigma_range than
// shadow bilateral: smoothes near-clipped specular without touching midtones.
static struct pl_hook_res highlight_bilateral_hook(void *priv,
                                                   const struct pl_hook_params *params)
{
    struct pl_ml_render_result *result = priv;
    bool has_rolloff = (result->highlight_rolloff > 0.005f ||
                        result->highlight_rolloff < -0.005f);
    if (result->highlight_strength <= 0.001f && !has_rolloff)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    pl_tex src = params->tex;
    int w = src->params.w, h = src->params.h;
    pl_tex dst = params->get_tex(params->priv, w, h);
    if (!dst)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    pl_shader sh = pl_dispatch_begin(params->dispatch);

    struct pl_shader_var vars[] = {
        { .var = pl_var_float("hi_rolloff"), .data = &result->highlight_rolloff,  .dynamic = true },
        { .var = pl_var_float("hi_str"),     .data = &result->highlight_strength, .dynamic = true },
        { .var = pl_var_float("hi_knee"),    .data = &result->highlight_knee,     .dynamic = true },
    };
    struct pl_shader_desc desc = {
        .desc = { .name = "hi_src", .type = PL_DESC_SAMPLED_TEX },
        .binding = { .object = src,
                     .address_mode = PL_TEX_ADDRESS_CLAMP,
                     .sample_mode  = PL_TEX_SAMPLE_NEAREST },
    };
    // Highlight recovery: two-stage pass.
    // Stage 1 — shoulder roll-off: above the knee, blend toward a soft shoulder.
    //   hi_rolloff > 0 → boost highlights (expand), < 0 → crush (compress).
    //   Quadratic: out = y + rolloff * ((y - knee) / (1 - knee))^2 * (1 - y)
    //   Anchored at knee (no change) and at white (no change), smooth.
    // Stage 2 — bilateral smoothing (existing): softens clipped specular.
    static const char highlight_body[] =
        "vec2 sz = vec2(textureSize(hi_src, 0));\n"
        "vec2 uv = gl_FragCoord.xy / sz;\n"
        "vec2 pt = 1.0 / sz;\n"
        "vec4 csrc = textureLod(hi_src, uv, 0.0);\n"
        "vec3 center = csrc.rgb;\n"
        "float y_c = dot(center, vec3(0.2126, 0.7152, 0.0722));\n"
        "vec3 out_rgb = center;\n"
        "if (abs(hi_rolloff) > 0.005 && y_c > hi_knee && y_c < 0.995) {\n"
        "    float span = max(1.0 - hi_knee, 0.01);\n"
        "    float t = (y_c - hi_knee) / span;\n"
        "    float shoulder = hi_rolloff * t * t * (1.0 - y_c);\n"
        "    float y_new = clamp(y_c + shoulder, 0.0, 1.0);\n"
        "    float ratio_hi = (y_c > 0.001) ? y_new / y_c : 1.0;\n"
        "    out_rgb = clamp(center * ratio_hi, 0.0, 1.0);\n"
        "    y_c = y_new;\n"
        "}\n"
        "float hmask = smoothstep(hi_knee * 0.9, hi_knee, y_c);\n"
        "hmask *= smoothstep(0.0, 3.0 / sz.y, min(uv.y, 1.0 - uv.y));\n"
        "if (hmask > 0.001 && hi_str > 0.001) {\n"
        "    vec3 acc = vec3(0.0); float tw = 0.0;\n"
        "    float sr = clamp(hi_str * 0.5, 0.01, 0.15);\n"
        "    for (int dy = -2; dy <= 2; dy++) {\n"
        "        for (int dx = -2; dx <= 2; dx++) {\n"
        "            vec3 s = textureLod(hi_src, uv + vec2(dx, dy) * pt, 0.0).rgb;\n"
        "            float sy = dot(s, vec3(0.2126, 0.7152, 0.0722));\n"
        "            float d2 = float(dx*dx + dy*dy);\n"
        "            float yd = y_c - sy;\n"
        "            float w = exp(-d2 * 0.222) * exp(-yd*yd / (2.0*sr*sr));\n"
        "            acc += w * s; tw += w;\n"
        "        }\n"
        "    }\n"
        "    out_rgb = mix(out_rgb, acc / max(tw, 0.001),\n"
        "                  hmask * clamp(hi_str * 2.5, 0.0, 1.0));\n"
        "}\n"
        "color = vec4(out_rgb, csrc.a);\n";

    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "Highlight roll-off + bilateral smoothing",
        .body        = highlight_body,
        .input       = PL_SHADER_SIG_NONE,
        .output      = PL_SHADER_SIG_COLOR,
        .variables   = vars, .num_variables   = 3,
        .descriptors = &desc, .num_descriptors = 1,
        .output_w    = w,     .output_h        = h,
    })) {
        pl_dispatch_abort(params->dispatch, &sh);
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };
    }
    if (!pl_dispatch_finish(params->dispatch,
            pl_dispatch_params(.shader = &sh, .target = dst)))
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    return (struct pl_hook_res) {
        .output     = PL_HOOK_SIG_TEX,
        .tex        = dst,
        .repr       = params->repr,
        .color      = params->color,
        .components = params->components,
        .rect       = params->rect,
    };
}

// Construct CR — log-space Gaussian unsharp mask.
//
// Single SIG_TEX hook that runs after the Oracle (l2_hook).  Extracts
// per-pixel log-luma detail by subtracting a Gaussian-blurred log-luma
// field, then adds the detail back scaled by cr_str.  This restores the
// local contrast that the Oracle gamma flattened without needing cross-hook
// state (the alpha-channel approach is rejected at PL_HOOK_OUTPUT because
// libplacebo disallows component-count changes at non-resizable stages).
//
// When Construct CR is active, native libplacebo contrast_recovery is
// zeroed in vo_gpu_next to prevent double processing.

static struct pl_hook_res cr_unsharpmask_hook(void *priv,
                                              const struct pl_hook_params *params)
{
    struct pl_ml_render_result *result = priv;
    if (result->cr_strength <= 0.001f)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    pl_tex src = params->tex;
    int w = src->params.w, h = src->params.h;
    pl_tex dst = params->get_tex(params->priv, w, h);
    if (!dst)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    pl_shader sh = pl_dispatch_begin(params->dispatch);

    struct pl_shader_var vars[] = {
        { .var = pl_var_float("cr_str"), .data = &result->cr_strength, .dynamic = true },
    };
    struct pl_shader_desc desc = {
        .desc = { .name = "cr_src", .type = PL_DESC_SAMPLED_TEX },
        .binding = { .object = src,
                     .address_mode = PL_TEX_ADDRESS_CLAMP,
                     .sample_mode  = PL_TEX_SAMPLE_NEAREST },
    };

    // Two-scale log-space Gaussian unsharp mask:
    // 5×5 Gaussian kernel, stride 4 (~16 px effective radius at 4K).
    // Operates in log-luma space for perceptual uniformity.
    static const char cr_body[] =
        "vec2 sz = vec2(textureSize(cr_src, 0));\n"
        "vec2 uv = gl_FragCoord.xy / sz;\n"
        "vec2 pt = 1.0 / sz;\n"
        "vec3 center = textureLod(cr_src, uv, 0.0).rgb;\n"
        "float y_c = dot(center, vec3(0.2126, 0.7152, 0.0722));\n"
        "float eps = 1e-4;\n"
        "vec3 out_rgb = center;\n"
        "float cr_edge = smoothstep(0.0, 9.0 / sz.y, min(uv.y, 1.0 - uv.y))\n"
        "              * smoothstep(0.0, 9.0 / sz.x, min(uv.x, 1.0 - uv.x));\n"
        "if (y_c > 0.02 && cr_edge > 0.001) {\n"
        "    float log_yc = log(y_c + eps);\n"
        "    float blur_acc = 0.0; float tw = 0.0;\n"
        "    for (int dy = -2; dy <= 2; dy++) {\n"
        "        for (int dx = -2; dx <= 2; dx++) {\n"
        "            vec3 s = textureLod(cr_src, uv + vec2(dx,dy) * pt * 4.0, 0.0).rgb;\n"
        "            float sy = dot(s, vec3(0.2126, 0.7152, 0.0722));\n"
        "            if (sy > 0.02) {\n"
        "                float d2 = float(dx*dx + dy*dy);\n"
        "                float w = exp(-d2 * 0.25);\n"
        "                blur_acc += w * log(sy + eps);\n"
        "                tw += w;\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "    float log_blur = blur_acc / max(tw, 0.001);\n"
        "    float detail = log_yc - log_blur;\n"
        "    float hi_sup = 1.0 - smoothstep(0.85, 1.0, y_c);\n"
        "    float lo_sup = smoothstep(0.01, 0.05, y_c);\n"
        "    float log_new = log_yc + cr_str * detail * hi_sup * lo_sup * cr_edge;\n"
        "    float y_new = clamp(exp(log_new) - eps, 0.0, 1.0);\n"
        "    float ratio = y_new / y_c;\n"
        "    out_rgb = clamp(center * ratio, 0.0, 1.0);\n"
        "}\n"
        "color = vec4(out_rgb, 1.0);\n";

    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "Contrast recovery (log-space unsharp mask)",
        .body        = cr_body,
        .input       = PL_SHADER_SIG_NONE,
        .output      = PL_SHADER_SIG_COLOR,
        .variables   = vars, .num_variables   = 1,
        .descriptors = &desc, .num_descriptors = 1,
        .output_w    = w,     .output_h        = h,
    })) {
        pl_dispatch_abort(params->dispatch, &sh);
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };
    }
    if (!pl_dispatch_finish(params->dispatch,
            pl_dispatch_params(.shader = &sh, .target = dst)))
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    return (struct pl_hook_res) {
        .output     = PL_HOOK_SIG_TEX,
        .tex        = dst,
        .repr       = params->repr,
        .color      = params->color,
        .components = params->components,
        .rect       = params->rect,
    };
}

static bool build_model_features(const struct pl_ml_render_params *params,
                                 float features[88])
{
    // Read the 78-element base feature vector (features[0–77]) from the peak
    // detection buffer accumulated during the most recent pl_render_image call.
    // No GPU render pass is issued here — zero marginal cost.
    if (!pl_renderer_get_ml_features(params->renderer,
                                     params->target_nits, features))
        return false;

    // Append 10 spline-knot features (features[78–87]) built from the current
    // libplacebo tone-map spline — same generator the model was trained on.
    struct pl_tone_map_params spline = {
        .function = &pl_tone_map_spline,
        .constants = { PL_TONE_MAP_CONSTANTS },
        .input_scaling = PL_HDR_PQ,
        .output_scaling = PL_HDR_PQ,
        .lut_size = 256,
        .output_max = 0.5444f,
    };
    if (params->is_sdr && !params->emulate_sdr) {
        // SDR (NOT emulated): feed the spline a *virtual DV-mastered* input so
        // the knots (77–84) encode the curve a Dolby L2 master would write for
        // this frame, while staying in the trained LUT-sample (output-PQ)
        // units. Base stats are already true PQ-of-nits from the shader fix.
        // APL sigmoid gain centered on ~18-nit SDR mid-gray; virtual ceiling
        // fixed at 1000 nits (0.7518 PQ) per the SDR design.
        // Under the SDR→P5 emulation bridge the signal is already virtual-HDR,
        // so this branch is bypassed and the natural spline takes over.
        float nits_max = sdr_pq_to_nits(features[0]);
        float nits_avg = sdr_pq_to_nits(features[1]);
        float sigma = 1.0f / (1.0f + expf(-0.15f * (nits_avg - 18.0f)));
        float g_avg = 1.0f + 1.50f * (1.0f - sigma);
        spline.input_avg = sdr_nits_to_pq(nits_avg * g_avg);
        spline.input_max = fminf(sdr_nits_to_pq(nits_max * 2.5f),
                                 SDR_VIRTUAL_PEAK_PQ);
        spline.input_max = fmaxf(spline.input_max, spline.input_avg + 1e-3f);
    } else {
        spline.input_max = fmaxf(params->l1_max_pq, features[0]);
        spline.input_avg = params->l1_avg_pq > 0.0f ? params->l1_avg_pq : features[1];
    }
    float spline_lut[256];
    const int knot_indices[8] = { 0, 36, 73, 109, 146, 182, 219, 255 };
    pl_tone_map_params_infer(&spline);
    pl_tone_map_generate(spline_lut, &spline);
    // features[77..84]: spline knots — overwrites target_nits slot, matching
    // the layout the trained XGBoost model expects (88-feature input vector).
    for (int index = 0; index < 8; index++)
        features[77 + index] = spline_lut[knot_indices[index]];
    features[85] = params->top_bar_norm;
    features[86] = params->bottom_bar_norm;
    // Feature 87 (compression_strength): ratio of output target to scene peak.
    // SDR: floor features[0] at 0.62 so 0.5444/0.51 never reaches the model.
    // HDR: soft cap at 0.90 PQ (~4000 nits) catches extreme container peaks
    // without restricting the model's view of normal high-nit content.
    float f0_den;
    if (params->is_sdr && !params->emulate_sdr) {
        f0_den = fmaxf(features[0], SDR_FEATURE0_FLOOR);
    } else {
        f0_den = fminf(fmaxf(features[0], 1e-6f), 0.90f);
    }
    features[87] = 0.5444f / f0_den;
    return true;
}

bool pl_ml_render_evaluate(const struct pl_ml_render_params *params,
                           struct pl_ml_render_result *result)
{
    if (!params || !result)
        return false;

    float features[88];
    if (!build_model_features(params, features))
        return false;

    *result = (struct pl_ml_render_result) {
        .l1_max_pq = params->l1_max_pq > 0.0f ? params->l1_max_pq : features[0],
        .l1_avg_pq = params->l1_avg_pq > 0.0f ? params->l1_avg_pq : features[1],
        .fire_pop_strength = params->fire_pop_mode == PL_ML_CONTROL_MANUAL ?
            params->fire_pop_strength : 0.0f,
    };

    // Forward the skin LUT to the result (wiped by the assignment above).
    // Unconditional: the chroma hook may run whenever chroma boost > 1.0, in
    // any control mode. When NULL (no LUT configured), the hook falls back to
    // the legacy ellipse body.
    result->skin_lut     = params->skin_lut;
    result->skin_lut_gpu = params->skin_lut_gpu;
    result->skin_lut_cr0 = params->skin_lut_cr0;
    result->skin_lut_cr1 = params->skin_lut_cr1;
    result->skin_lut_cb0 = params->skin_lut_cb0;
    result->skin_lut_cb1 = params->skin_lut_cb1;
    result->hue_correction = params->hue_correction;

    struct pl_ml_prediction prediction;
    float gamma = 1.0f;
    if (params->gamma_mode == PL_ML_CONTROL_AUTO && params->model) {
        if (pl_ml_context_predict(params->model, features, 88, &prediction)) {
            gamma = prediction.gamma;
            result->model_used = true;
        } else {
            result->model_fallback = true;
        }
    } else if (params->gamma_mode == PL_ML_CONTROL_MANUAL) {
        gamma = params->gamma;
    }
    result->gamma = fmaxf(0.5f, fminf(1.5f, gamma));
    result->l2_power = 2048.0f / result->gamma;
    result->l2_saturation = 2048.0f;
    // Adaptive highlight guard: lower the L2 shader's highlight protection
    // threshold for bright scenes so sky/specular pixels are protected while
    // dark/mid areas still get full contrast expansion from the gamma curve.
    {
        float l1max = params->l1_max_pq > 0.0f ? params->l1_max_pq : features[0];
        result->l2_highlight_guard = 0.85f;
        if (l1max > 0.6f)
            result->l2_highlight_guard = fmaxf(0.50f,
                0.85f - (l1max - 0.6f) * 0.75f);
        result->hunt_strength = params->hunt_compensation;
        result->highlight_desat = params->highlight_desat;
    }

    if (params->cr_mode == PL_ML_CONTROL_MANUAL) {
        result->cr_strength = params->cr_strength;
    } else if (params->cr_mode == PL_ML_CONTROL_AUTO) {
        float base_cr = fmaxf(0.1f, fminf(0.5f,
            0.25f + (1.2f - result->gamma) * 0.15f));
        // Taper CR for very bright/high-contrast scenes (fires, explosions,
        // bright sky). At l1_max > 0.7 the bilateral filter has extreme
        // gradients that crush bright edges. Scale back up to 80% reduction
        // at l1_max=1.0 to prevent compounding with gamma and radiance boosts.
        float l1max = result->l1_max_pq > 0.0f ? result->l1_max_pq :
                      (params->l1_max_pq > 0.0f ? params->l1_max_pq : 0.5f);
        float brightness_taper = 1.0f - fmaxf(0.0f, (l1max - 0.7f) / 0.3f) * 0.8f;
        result->cr_strength = base_cr * brightness_taper;
    }

    // Dehaze: shadow crush — steepens near-black ramp for depth.
    if (params->dehaze >= 0.0f) {
        result->dehaze_strength = fminf(1.0f, params->dehaze);
    } else {
        float l1 = result->l1_max_pq > 0.0f ? result->l1_max_pq : 0.5f;
        float darkness = fmaxf(0.0f, 1.0f - l1 / 0.85f);
        result->dehaze_strength = fminf(1.0f, 0.30f + darkness * 0.50f);
    }
    result->dehaze_knee = 0.30f;

    struct pl_ml_radiance_params radiance = params->radiance;
    radiance.average_luma = features[1];
    radiance.peak_luma = result->l1_max_pq > 0.0f ? result->l1_max_pq : features[0];
    pl_ml_radiance_configure(&result->radiance, &radiance);

    if (params->chroma_mode == PL_ML_CONTROL_MANUAL) {
        result->chroma_neutral_boost = params->chroma_neutral_boost;
        result->chroma_fire_boost    = params->chroma_fire_boost;
        result->chroma_knee          = params->chroma_knee;
        result->chroma_skin_protect  = params->chroma_skin_protect;
    } else if (params->chroma_mode == PL_ML_CONTROL_AUTO) {
        result->chroma_neutral_boost = 1.20f;
        result->chroma_fire_boost    = 1.35f;
        result->chroma_knee          = 0.55f;
        result->chroma_skin_protect  = 0.95f;
    } else {
        result->chroma_neutral_boost = 1.0f;
        result->chroma_fire_boost    = 1.0f;
        result->chroma_knee          = 0.55f;
        result->chroma_skin_protect  = 1.0f;
    }

    // ── SDR→P5 virtual-master bridge ─────────────────────────────────────────
    result->sdr_emulate = params->is_sdr && params->emulate_sdr;
    result->sdr_virtual_nits = params->sdr_virtual_nits > 0.0f
                             ? params->sdr_virtual_nits : 1000.0f;
    // APL-adaptive emulation strength (Dolby-flavoured): bright SDR scenes sit
    // near the virtual ceiling and need a gentle touch; dark scenes benefit
    // from a stronger mid/high lift. σ centred on ~18-nit SDR mid-gray.
    // features[1] is the true-PQ APL of the previous frame — 1-frame latency,
    // consistent with the rest of the ML path.
    float sdr_str = fmaxf(0.0f, fminf(1.0f, params->sdr_strength));
    if (result->sdr_emulate) {
        float avg_pq = features[1];
        float nits_avg = sdr_pq_to_nits(avg_pq);
        float sigma = 1.0f / (1.0f + expf(-0.15f * (nits_avg - 18.0f)));
        float adaptive = 1.0f + 1.0f * (1.0f - sigma);   // dark=2.0, bright=1.0
        sdr_str = fmaxf(0.0f, fminf(1.0f, sdr_str * adaptive));
    }
    result->sdr_strength = sdr_str;

    // ── Shadow recovery ─────────────────────────────────────────────────────
    result->shadow_knee = 0.25f;
    result->shadow_toe = 0.0f;
    result->shadow_strength = 0.0f;
    if (params->shadow_mode == PL_ML_CONTROL_MANUAL) {
        result->shadow_toe = fmaxf(0.0f, fminf(0.5f, params->shadow_strength));
    } else if (params->shadow_mode == PL_ML_CONTROL_AUTO) {
        if (params->shadow_model) {
            struct pl_ml_prediction shadow_pred;
            if (pl_ml_context_predict(params->shadow_model, features, 88, &shadow_pred))
                result->shadow_toe = fmaxf(0.0f, fminf(0.5f, shadow_pred.value));
        }
        // Bilateral detail injection: heuristic from scene darkness.
        float avg = features[1] > 0.0f ? features[1] : 0.15f;
        result->shadow_strength = fmaxf(0.0f, fminf(0.30f, (0.30f - avg) * 0.8f));
    }

    // ── Highlight recovery ──────────────────────────────────────────────────
    // Knee tracks Oracle gamma: brighter/flatter curve → lower compression
    // point → knee shifts to match where the curve starts rolling off.
    float safe_gamma = fmaxf(result->gamma, 0.5f);
    result->highlight_knee = fmaxf(0.65f,
                                   fminf(0.85f, 0.90f - (1.0f / safe_gamma) * 0.20f));
    result->highlight_rolloff = 0.0f;
    result->highlight_strength = 0.0f;
    if (params->highlight_mode == PL_ML_CONTROL_MANUAL) {
        result->highlight_strength = fmaxf(0.0f, fminf(0.4f, params->highlight_strength));
    } else if (params->highlight_mode == PL_ML_CONTROL_AUTO) {
        if (params->highlight_model) {
            struct pl_ml_prediction hi_pred;
            if (pl_ml_context_predict(params->highlight_model, features, 88, &hi_pred))
                result->highlight_rolloff = fmaxf(-0.30f, fminf(0.40f, hi_pred.value));
        }
        // Bilateral smoothing: heuristic from scene peak brightness.
        float peak = features[0] > 0.0f ? features[0] : 0.5f;
        result->highlight_strength = fmaxf(0.0f, fminf(0.25f, (peak - 0.55f) * 0.6f));
    }

    // ── Highlight headroom ───────────────────────────────────────────────────
    if (params->headroom >= 0.0f) {
        result->headroom = fminf(0.30f, fmaxf(0.0f, params->headroom));
    } else {
        float boost = 0.0f;
        if (result->fire_pop_strength > 0.0f)
            boost += 0.04f;
        if (result->l2_power != 2048.0f) {
            float g = 2048.0f / fmaxf(result->l2_power, 1.0f);
            boost += fmaxf(0.0f, 1.0f - g) * 0.15f;
        }
        if (result->cr_strength > 0.001f)
            boost += result->cr_strength * 0.10f;
        if (result->radiance.strength > 0.0f)
            boost += result->radiance.strength * 0.06f;
        result->headroom = fminf(0.15f, fmaxf(0.0f, boost));
    }

    // ── HDR texture recovery + vector desaturation ──────────────────────────
    result->hdr_detail_strength = fmaxf(0.0f, fminf(1.0f, params->hdr_detail));
    result->vector_desat_strength = fmaxf(0.0f, fminf(0.6f, params->vector_desat));

    // ── Two-stage tone mapping ──────────────────────────────────────────────
    result->dc_ceiling_pq = 0.0f;
    result->dc_compress_gain = 0.0f;
    if (params->virtual_target_nits > 0.0f &&
        params->virtual_target_nits < 10000.0f &&
        params->target_nits > 0.0f &&
        params->virtual_target_nits > params->target_nits)
    {
        result->dc_compress_gain =
            params->virtual_target_nits / params->target_nits;
    }

    return true;
}

// ── Vector Channel Desaturation (PL_HOOK_NATIVE, PQ domain) ─────────────
// Trades highlight colour saturation for tonal headroom before tone mapping.
// By pulling near-clip RGB channels toward their luma anchor, inter-channel
// gradients survive the subsequent compression — preserving texture in clouds,
// sand, specular surfaces.
static struct pl_hook_res vector_desat_hook(void *priv,
                                            const struct pl_hook_params *params)
{
    struct pl_ml_render_result *result = priv;
    if (result->vector_desat_strength <= 0.001f)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };
    if (params->color.transfer != PL_COLOR_TRC_PQ)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    pl_shader sh = params->sh;
    struct pl_shader_var vars[] = {
        { .var = pl_var_float("vd_str"), .data = &result->vector_desat_strength, .dynamic = true },
    };

    static const char body[] =
        "vec3 rgb = max(color.rgb, vec3(0.0));\n"
        "float y_pq = dot(rgb, vec3(0.2126, 0.7152, 0.0722));\n"
        "float desat = smoothstep(0.55, 0.85, y_pq);\n"
        "if (desat > 0.001) {\n"
        "    color.rgb = mix(rgb, vec3(y_pq), desat * vd_str);\n"
        "}\n";

    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "Vector channel desaturation",
        .body        = body,
        .input       = PL_SHADER_SIG_COLOR,
        .output      = PL_SHADER_SIG_COLOR,
        .variables   = vars, .num_variables = 1,
    })) {
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };
    }
    return (struct pl_hook_res) {
        .output = PL_HOOK_SIG_COLOR, .sh = sh,
        .repr = params->repr, .color = params->color,
        .components = params->components, .rect = params->rect,
    };
}

// ── HDR Detail Enhance (PL_HOOK_NATIVE, PQ domain) ──────────────────────
// Amplifies high-frequency luma texture in the PQ signal BEFORE tone mapping.
// A 5×5 gaussian highpass isolates structural gradients (clouds, sand, specular)
// and adds them back with gain in the highlight region.  The tone mapper then
// compresses an input with exaggerated texture — the gradients survive even
// aggressive mapping to 50 nits, matching madVR's texture-injection philosophy.
static struct pl_hook_res hdr_detail_enhance_hook(void *priv,
                                                   const struct pl_hook_params *params)
{
    struct pl_ml_render_result *result = priv;
    if (result->hdr_detail_strength <= 0.001f)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };
    if (params->color.transfer != PL_COLOR_TRC_PQ)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    pl_tex src = params->tex;
    int w = src->params.w, h = src->params.h;
    pl_tex dst = params->get_tex(params->priv, w, h);
    if (!dst)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    pl_shader sh = pl_dispatch_begin(params->dispatch);

    struct pl_shader_var vars[] = {
        { .var = pl_var_float("det_str"), .data = &result->hdr_detail_strength, .dynamic = true },
    };
    struct pl_shader_desc desc = {
        .desc = { .name = "hdr_enh_src", .type = PL_DESC_SAMPLED_TEX },
        .binding = { .object = src,
                     .address_mode = PL_TEX_ADDRESS_CLAMP,
                     .sample_mode  = PL_TEX_SAMPLE_NEAREST },
    };

    static const char enhance_body[] =
        "vec2 sz = vec2(textureSize(hdr_enh_src, 0));\n"
        "vec2 uv = gl_FragCoord.xy / sz;\n"
        "vec2 pt = 1.0 / sz;\n"
        "vec4 csrc = textureLod(hdr_enh_src, uv, 0.0);\n"
        "vec3 center = csrc.rgb;\n"
        "float y_c = dot(center, vec3(0.2126, 0.7152, 0.0722));\n"
        "float y_smooth = 0.0; float tw = 0.0;\n"
        "for (int dy = -2; dy <= 2; dy++) {\n"
        "    for (int dx = -2; dx <= 2; dx++) {\n"
        "        vec3 s = textureLod(hdr_enh_src, uv + vec2(dx,dy) * pt, 0.0).rgb;\n"
        "        float sy = dot(s, vec3(0.2126, 0.7152, 0.0722));\n"
        "        float d2 = float(dx*dx + dy*dy);\n"
        "        float w = exp(-d2 * 0.125);\n"
        "        y_smooth += w * sy; tw += w;\n"
        "    }\n"
        "}\n"
        "y_smooth /= max(tw, 0.001);\n"
        "float detail = y_c - y_smooth;\n"
        // Enhance highlights: amplify texture where tone mapping will compress most.
        // Mask ramps from 0 at PQ 0.45 (~60 nits) to full at PQ 0.75 (~700 nits).
        // Shadow/midtone detail is already well-preserved by the tone mapper.
        "float hi_mask = smoothstep(0.45, 0.75, y_c);\n"
        "float gain = det_str * hi_mask * 3.0;\n"
        "float y_new = y_c + detail * gain;\n"
        // Ratio-preserving RGB scaling (like headroom hook)
        "vec3 out_rgb = center;\n"
        "if (y_c > 0.001) out_rgb *= y_new / y_c;\n"
        "color = vec4(clamp(out_rgb, 0.0, 1.0), csrc.a);\n";

    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "HDR detail enhance",
        .body        = enhance_body,
        .input       = PL_SHADER_SIG_NONE,
        .output      = PL_SHADER_SIG_COLOR,
        .variables   = vars, .num_variables   = 1,
        .descriptors = &desc, .num_descriptors = 1,
        .output_w    = w, .output_h = h,
    })) {
        pl_dispatch_abort(params->dispatch, &sh);
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };
    }
    if (!pl_dispatch_finish(params->dispatch,
            pl_dispatch_params(.shader = &sh, .target = dst)))
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    return (struct pl_hook_res) {
        .output     = PL_HOOK_SIG_TEX,
        .tex        = dst,
        .repr       = params->repr,
        .color      = params->color,
        .components = params->components,
        .rect       = params->rect,
    };
}

int pl_ml_render_get_hooks(struct pl_ml_render_result *result,
                           struct pl_hook *hooks)
{
    if (!result || !hooks)
        return 0;
    int count = 0;
    // SDR→P5 emulation runs before everything else at NATIVE so the shadow
    // mask and every downstream stage see the virtual-PQ (DV-mastered) signal.
    // NOTE: PL_HOOK_NATIVE fires after YUV→RGB merge, signal still in PQ.
    // PL_HOOK_RGB_INPUT only fires for RGB-format input (not YUV content).
    if (result->sdr_emulate) {
        hooks[count++] = (struct pl_hook) {
            .stages = PL_HOOK_NATIVE,
            .input  = PL_HOOK_SIG_TEX,
            .priv   = result,
            .hook   = sdr_p5_hook,
            .signature = 0x5344525035465348ull,  // "SDRP5SH"
        };
    }
    if (result->dc_ceiling_pq > 0.001f) {
        hooks[count++] = (struct pl_hook) {
            .stages = PL_HOOK_NATIVE,
            .input  = PL_HOOK_SIG_COLOR,
            .priv   = result,
            .hook   = pq_precompress_hook,
            .signature = 0x5051505245434D50ull,  // "PQPRECMP"
        };
    }
    if (result->vector_desat_strength > 0.001f) {
        hooks[count++] = (struct pl_hook) {
            .stages = PL_HOOK_NATIVE,
            .input  = PL_HOOK_SIG_COLOR,
            .priv   = result,
            .hook   = vector_desat_hook,
            .signature = 0x5645434453415448ull,  // "VECDSATH"
        };
    }
    if (result->hdr_detail_strength > 0.001f) {
        hooks[count++] = (struct pl_hook) {
            .stages = PL_HOOK_NATIVE,
            .input  = PL_HOOK_SIG_TEX,
            .priv   = result,
            .hook   = hdr_detail_enhance_hook,
            .signature = 0x484452454E484E48ull,  // "HDRENHN"
        };
    }
    // Display compress + headroom: not registered. The virtual peak override
    // alone shapes the tone map curve for highlight preservation; a post-TM
    // compress at gain=4x lifts midtones unacceptably on low-peak displays.
    // The hook functions and output_w/output_h fixes are retained for future use.
    (void) display_compress_hook;
    (void) headroom_hook;
    // OUTPUT hook order: shadow → highlight → gamma → CR → radiance → chroma
    // Shadow toe-lift + bilateral: first so toe-lift sets the black level
    // before gamma's fade-in region activates.
    if (result->shadow_toe > 0.001f || result->shadow_strength > 0.001f) {
        hooks[count++] = (struct pl_hook) {
            .stages = PL_HOOK_OUTPUT,
            .input  = PL_HOOK_SIG_TEX,
            .priv   = result,
            .hook   = shadow_bilateral_hook,
            .signature = 0x4D4C53484244574Cull,  // "MLSHBDWL"
        };
    }
    // Highlight roll-off + bilateral: shapes the shoulder before gamma
    // so the highlight guard region is well-defined.
    {
        bool has_hi = (result->highlight_rolloff > 0.005f ||
                       result->highlight_rolloff < -0.005f ||
                       result->highlight_strength > 0.001f);
        if (has_hi) {
            hooks[count++] = (struct pl_hook) {
                .stages = PL_HOOK_OUTPUT,
                .input  = PL_HOOK_SIG_TEX,
                .priv   = result,
                .hook   = highlight_bilateral_hook,
                .signature = 0x4D4C48494C545348ull,  // "MLHILTSH"
            };
        }
    }
    if (result->fire_pop_strength > 0.0f) {
        hooks[count++] = (struct pl_hook) { .stages = PL_HOOK_OUTPUT,
            .input = PL_HOOK_SIG_COLOR, .priv = result, .hook = fire_hook,
            .signature = 0x4456504649524550ull };
    }
    // L2 gamma + saturation: after shadow/highlight shaping.
    if (result->l2_power != 2048.0f || result->l2_saturation != 2048.0f) {
        hooks[count++] = (struct pl_hook) { .stages = PL_HOOK_OUTPUT,
            .input = PL_HOOK_SIG_COLOR, .priv = result, .hook = l2_hook,
            .signature = 0x44564C325452494Dull };
    }
    // Construct CR: log-space Gaussian unsharp mask (runs after Oracle)
    if (result->cr_strength > 0.001f) {
        hooks[count++] = (struct pl_hook) {
            .stages = PL_HOOK_OUTPUT,
            .input  = PL_HOOK_SIG_TEX,
            .priv   = result,
            .hook   = cr_unsharpmask_hook,
            .signature = 0x435255534D41534Bull,  // "CRUSMASK"
        };
    }
    if (result->radiance.strength > 0.0f)
        pl_ml_radiance_get_hook(&result->radiance, &hooks[count++]);
    if (result->chroma_neutral_boost > 1.0f || result->chroma_fire_boost > 1.0f) {
        hooks[count++] = (struct pl_hook) { .stages = PL_HOOK_OUTPUT,
            .input = PL_HOOK_SIG_COLOR, .priv = result, .hook = chroma_tuner_hook,
            .signature = 0x4348524D41544E55ull };
    }
    return count;
}
