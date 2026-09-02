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
        "float l2_gamma = 2048.0 / l2_power;\n"
        "if (l2_gamma != 1.0 && y > 0.001 && y < 0.999) {\n"
        "    float hi = smoothstep(l2_guard, 0.98, y);\n"
        "    float g = mix(l2_gamma, 1.0, hi);\n"
        "    float lo = 0.5 * pow(2.0 * y, g);\n"
        "    float hi_v = 1.0 - 0.5 * pow(2.0 * (1.0 - y), g);\n"
        "    y = mix(lo, hi_v, step(0.5, y));\n"
        "}\n"
        "if (l2_midtone_boost > 0.001) {\n"
        "    float hi_mb = smoothstep(l2_guard, 0.98, y);\n"
        "    float eff_boost = mix(l2_midtone_boost, 0.0, hi_mb);\n"
        "    float shadow_bias = max(0.0, 0.5 - y) * eff_boost * 0.4;\n"
        "    y = clamp(0.5 + (y - 0.5) * (1.0 + eff_boost) - shadow_bias, 0.0, 1.0);\n"
        "}\n"
        "float desat = smoothstep(l2_guard, 0.95, y);\n"
        "cr *= (1.0 - desat * 0.5);\n"
        "cb *= (1.0 - desat * 0.5);\n"
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
        { .var = pl_var_float("l2_midtone_boost"), .data = &result->l2_midtone_boost, .dynamic = true },
    };
    ml_grade_to_ref(sh, &target_csp);
    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "GPU L2 gamma and saturation trim", .body = body,
        .input = PL_SHADER_SIG_COLOR, .output = PL_SHADER_SIG_COLOR,
        .variables = vars, .num_variables = 4,
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
        // 6. Warm-excursion damp — same counter-weight as body_b
        "float warm = clamp((r_res - b_res) * 3.3333, 0.0, 1.0);\n"
        "chroma_scalar = mix(chroma_scalar, 1.0, warm * u_chroma_warm_damp);\n"
        // 7. Apply scalar to residuals, reconstruct gamma-2.2 signal
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
        // 4b. Warm-excursion damp -- the red/pink counter-weight
        "float warm = clamp((r_res - b_res) * 3.3333, 0.0, 1.0);\n"
        "chroma_scalar = mix(chroma_scalar, 1.0, warm * u_chroma_warm_damp);\n"
        // 5. Apply scalar to residuals, reconstruct gamma-2.2 signal
        "r_res *= chroma_scalar; b_res *= chroma_scalar;\n"
        "color.r = clamp(y + r_res, 0.0, 1.0);\n"
        "color.g = clamp(y - (0.2126 / 0.7152) * r_res - (0.0722 / 0.7152) * b_res, 0.0, 1.0);\n"
        "color.b = clamp(y + b_res, 0.0, 1.0);\n";
    struct pl_shader_var vars[] = {
        { .var = pl_var_float("u_chroma_neutral_boost"), .data = &result->chroma_neutral_boost, .dynamic = true },
        { .var = pl_var_float("u_chroma_fire_boost"),    .data = &result->chroma_fire_boost,    .dynamic = true },
        { .var = pl_var_float("u_chroma_knee"),          .data = &result->chroma_knee,          .dynamic = true },
        { .var = pl_var_float("u_chroma_skin_protect"),  .data = &result->chroma_skin_protect,  .dynamic = true },
        // warm-excursion damp [0..1] — shared by both body-A and body-B
        { .var = pl_var_float("u_chroma_warm_damp"), .data = &result->chroma_warm_damp, .dynamic = true },
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
        .variables = vars, .num_variables = lut_ok ? 9 : 5,
        .descriptors = lut_ok ? &desc : NULL,
        .num_descriptors = lut_ok ? 1 : 0,
    })) return (struct pl_hook_res) { .failed = true };
    ml_grade_from_ref(sh, &target_csp);
    return (struct pl_hook_res) { .output = PL_HOOK_SIG_COLOR, .sh = sh,
        .repr = params->repr, .color = params->color, .components = params->components,
        .rect = params->rect };
}

// Shadow bilateral — PL_HOOK_RGB_INPUT (PQ signal, pre-linearization).
// 5×5 bilateral filter active where y_pq < shadow_knee (≈ 11.7 nits).
// Sigma-spatial: fixed 1.5 px.  Sigma-range: from shadow_strength.
// Uses PL_HOOK_SIG_TEX I/O: dispatches its own full-image render pass so it
// can sample neighboring pixels (not possible with PL_HOOK_SIG_COLOR).
//
// The learned toe un-crushes dark gradients via a multiplicative gain
//   out = rgb * (1 + str * (1 - y_pq/knee)^2)   (y_pq < knee)
// anchored at black (out(0)=0 → no fog/veil), slope at black = 1+str > 1
// (near-black detail contrast *expands*), identity with slope 1 at the knee
// (C1-continuous, monotonic).  A constant additive offset in PQ code space
// — as used before — lifted black to a gray veil and *compressed* shadow
// gradients, the exact opposite of un-crushing.
static struct pl_hook_res shadow_bilateral_hook(void *priv,
                                                const struct pl_hook_params *params)
{
    struct pl_ml_render_result *result = priv;
    if (result->shadow_strength <= 0.001f)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    pl_tex src = params->tex;
    int w = src->params.w, h = src->params.h;
    pl_tex dst = params->get_tex(params->priv, w, h);
    if (!dst)
        return (struct pl_hook_res) { .failed = true };

    pl_shader sh = pl_dispatch_begin(params->dispatch);

    // BT.2020 for HDR sources, BT.709 for SDR (after SDR→P5 bridge)
    float luma[3] = { 0.2627f, 0.6780f, 0.0593f };
    if (result->sdr_emulate) {
        luma[0] = 0.2126f; luma[1] = 0.7152f; luma[2] = 0.0722f;
    }

    struct pl_shader_var vars[] = {
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
    // exp(-d2 * 0.222) = spatial Gaussian, sigma=1.5 px (1/(2*1.5^2) = 0.222)
    static const char shadow_body[] =
        "vec2 sz = vec2(textureSize(shadow_src, 0));\n"
        "vec2 uv = gl_FragCoord.xy / sz;\n"
        "vec2 pt = 1.0 / sz;\n"
        "vec4 csrc = textureLod(shadow_src, uv, 0.0);\n"
        "vec3 center = csrc.rgb;\n"
        "float y_c = dot(center, luma_w);\n"
        "float smask = 1.0 - smoothstep(shadow_knee * 0.5, shadow_knee, y_c);\n"
        "vec3 out_rgb = center;\n"
        "if (smask > 0.001 && shadow_str > 0.001) {\n"
        "    vec3 acc = vec3(0.0); float tw = 0.0;\n"
        "    float sr = clamp(shadow_str, 0.02, 0.30);\n"
        "    for (int dy = -2; dy <= 2; dy++) {\n"
        "        for (int dx = -2; dx <= 2; dx++) {\n"
        "            vec3 s = textureLod(shadow_src, uv + vec2(dx, dy) * pt, 0.0).rgb;\n"
        "            float sy = dot(s, luma_w);\n"
        "            float d2 = float(dx*dx + dy*dy);\n"
        "            float yd = y_c - sy;\n"
        "            float w = exp(-d2 * 0.222) * exp(-yd*yd / (2.0*sr*sr));\n"
        "            acc += w * s; tw += w;\n"
        "        }\n"
        "    }\n"
        "    vec3 denoised = acc / max(tw, 0.001);\n"
        "    out_rgb = mix(center, denoised, smask * clamp(shadow_str * 2.0, 0.0, 1.0));\n"
        "    float u = clamp(y_c / max(shadow_knee, 0.001), 0.0, 1.0);\n"
        "    float gain = 1.0 + shadow_str * (1.0 - u) * (1.0 - u);\n"
        "    out_rgb = clamp(mix(out_rgb, out_rgb * vec3(gain), smask), 0.0, 1.0);\n"
        "}\n"
        "color = vec4(out_rgb, csrc.a);\n";

    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "Shadow bilateral detail recovery",
        .body        = shadow_body,
        .input       = PL_SHADER_SIG_NONE,
        .output      = PL_SHADER_SIG_COLOR,
        .variables   = vars, .num_variables   = 3,
        .descriptors = &desc, .num_descriptors = 1,
        .output_w    = w,     .output_h        = h,
    })) {
        pl_dispatch_abort(params->dispatch, &sh);
        return (struct pl_hook_res) { .failed = true };
    }
    if (!pl_dispatch_finish(params->dispatch,
            pl_dispatch_params(.shader = &sh, .target = dst)))
        return (struct pl_hook_res) { .failed = true };

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
        return (struct pl_hook_res) { .failed = true };

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
        return (struct pl_hook_res) { .failed = true };
    }
    if (!pl_dispatch_finish(params->dispatch,
            pl_dispatch_params(.shader = &sh, .target = dst)))
        return (struct pl_hook_res) { .failed = true };

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
    if (result->highlight_strength <= 0.001f)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    pl_tex src = params->tex;
    int w = src->params.w, h = src->params.h;
    pl_tex dst = params->get_tex(params->priv, w, h);
    if (!dst)
        return (struct pl_hook_res) { .failed = true };

    pl_shader sh = pl_dispatch_begin(params->dispatch);

    struct pl_shader_var vars[] = {
        { .var = pl_var_float("hi_str"),  .data = &result->highlight_strength, .dynamic = true },
        { .var = pl_var_float("hi_knee"), .data = &result->highlight_knee,     .dynamic = true },
    };
    struct pl_shader_desc desc = {
        .desc = { .name = "hi_src", .type = PL_DESC_SAMPLED_TEX },
        .binding = { .object = src,
                     .address_mode = PL_TEX_ADDRESS_CLAMP,
                     .sample_mode  = PL_TEX_SAMPLE_NEAREST },
    };
    // Tighter range sigma (0.5× shadow) — specular highlight transitions are
    // abrupt; only blend across very similar luma values to preserve texture.
    static const char highlight_body[] =
        "vec2 sz = vec2(textureSize(hi_src, 0));\n"
        "vec2 uv = gl_FragCoord.xy / sz;\n"
        "vec2 pt = 1.0 / sz;\n"
        "vec4 csrc = textureLod(hi_src, uv, 0.0);\n"
        "vec3 center = csrc.rgb;\n"
        "float y_c = dot(center, vec3(0.2126, 0.7152, 0.0722));\n"
        "float hmask = smoothstep(hi_knee * 0.9, hi_knee, y_c);\n"
        "vec3 out_rgb = center;\n"
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
        "    out_rgb = mix(center, acc / max(tw, 0.001),\n"
        "                  hmask * clamp(hi_str * 2.5, 0.0, 1.0));\n"
        "}\n"
        "color = vec4(out_rgb, csrc.a);\n";

    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "Highlight bilateral smoothing",
        .body        = highlight_body,
        .input       = PL_SHADER_SIG_NONE,
        .output      = PL_SHADER_SIG_COLOR,
        .variables   = vars, .num_variables   = 2,
        .descriptors = &desc, .num_descriptors = 1,
        .output_w    = w,     .output_h        = h,
    })) {
        pl_dispatch_abort(params->dispatch, &sh);
        return (struct pl_hook_res) { .failed = true };
    }
    if (!pl_dispatch_finish(params->dispatch,
            pl_dispatch_params(.shader = &sh, .target = dst)))
        return (struct pl_hook_res) { .failed = true };

    return (struct pl_hook_res) {
        .output     = PL_HOOK_SIG_TEX,
        .tex        = dst,
        .repr       = params->repr,
        .color      = params->color,
        .components = params->components,
        .rect       = params->rect,
    };
}

// Construct CR — PL_HOOK_OUTPUT 9×9 bilateral for local contrast enhancement.
// Extracts the detail layer (original − bilateral) and boosts it by cr_strength.
// Replaces the native libplacebo contrast_recovery path to avoid the expensive
// get_feature_map downsample+extract pipeline.
static struct pl_hook_res cr_bilateral_hook(void *priv,
                                            const struct pl_hook_params *params)
{
    struct pl_ml_render_result *result = priv;
    if (result->cr_strength <= 0.001f)
        return (struct pl_hook_res) { .output = PL_HOOK_SIG_NONE };

    pl_tex src = params->tex;
    int w = src->params.w, h = src->params.h;
    pl_tex dst = params->get_tex(params->priv, w, h);
    if (!dst)
        return (struct pl_hook_res) { .failed = true };

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
    // 9×9 bilateral: spatial sigma=3.0 px (1/(2*9)≈0.0556), range sigma=0.10.
    // Luma-only detail boost — bilateral extracts the lowpass luma, difference
    // is the local contrast detail layer.  Chroma is preserved from the
    // original pixel (same principle as unsharp-mask in L*a*b*).
    static const char cr_body[] =
        "vec2 sz = vec2(textureSize(cr_src, 0));\n"
        "vec2 uv = gl_FragCoord.xy / sz;\n"
        "vec2 pt = 1.0 / sz;\n"
        "vec4 csrc = textureLod(cr_src, uv, 0.0);\n"
        "vec3 center = csrc.rgb;\n"
        "float y_c = dot(center, vec3(0.2126, 0.7152, 0.0722));\n"
        "float y_acc = 0.0; float tw = 0.0;\n"
        "float sr = 0.10;\n"
        "for (int dy = -4; dy <= 4; dy++) {\n"
        "    for (int dx = -4; dx <= 4; dx++) {\n"
        "        vec3 s = textureLod(cr_src, uv + vec2(dx, dy) * pt, 0.0).rgb;\n"
        "        float sy = dot(s, vec3(0.2126, 0.7152, 0.0722));\n"
        "        float d2 = float(dx*dx + dy*dy);\n"
        "        float yd = y_c - sy;\n"
        "        float w = exp(-d2 * 0.0556) * exp(-yd*yd / (2.0*sr*sr));\n"
        "        y_acc += w * sy; tw += w;\n"
        "    }\n"
        "}\n"
        "float y_bil = y_acc / max(tw, 0.001);\n"
        "float y_new = y_c + cr_str * (y_c - y_bil);\n"
        "y_new = clamp(y_new, 0.0, 1.0);\n"
        "color = vec4(clamp(center + (y_new - y_c), 0.0, 1.0), csrc.a);\n";

    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "Construct CR bilateral",
        .body        = cr_body,
        .input       = PL_SHADER_SIG_NONE,
        .output      = PL_SHADER_SIG_COLOR,
        .variables   = vars, .num_variables   = 1,
        .descriptors = &desc, .num_descriptors = 1,
        .output_w    = w,     .output_h        = h,
    })) {
        pl_dispatch_abort(params->dispatch, &sh);
        return (struct pl_hook_res) { .failed = true };
    }
    if (!pl_dispatch_finish(params->dispatch,
            pl_dispatch_params(.shader = &sh, .target = dst)))
        return (struct pl_hook_res) { .failed = true };

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
    result->chroma_warm_damp = params->chroma_warm_damp;

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
        // Compression-adaptive midtone contrast: the spline tone mapper
        // flattens midtone contrast proportional to how much it compresses.
        // Restore it with a midpoint stretch that scales with compression
        // ratio (peak / target). Uses the highlight guard so bright pixels
        // are excluded from the boost.
        float compression = l1max / 0.5444f;
        result->l2_midtone_boost = fmaxf(0.0f,
            fminf(0.80f, (compression - 1.0f) * 0.60f));
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

    // ── Shadow bilateral & toe lift ───────────────────────────────────────────
    result->shadow_knee = 0.35f; // fixed 0.35 PQ ≈ 11.7 nits shadow boundary
    result->shadow_strength = 0.0f;
    if (params->shadow_mode == PL_ML_CONTROL_MANUAL) {
        result->shadow_strength = fmaxf(0.0f, fminf(0.5f, params->shadow_strength));
    } else if (params->shadow_mode == PL_ML_CONTROL_AUTO) {
        // Model outputs a 0–0.4 metadata-derived target (base_score ≈ 0.044),
        // which sits below the shader's visible gain band. Scale it into the
        // 0–0.5 shader strength range so typical predictions actually un-crush.
        const float shadow_model_scale = 2.5f;
        if (params->shadow_model) {
            struct pl_ml_prediction shadow_pred;
            if (pl_ml_context_predict(params->shadow_model, features, 88, &shadow_pred))
                result->shadow_strength = fmaxf(0.0f, fminf(0.5f,
                           shadow_pred.value * shadow_model_scale));
        } else {
            // Heuristic: darker average scene → more shadow noise → more recovery.
            // features[1] ≈ l1_avg_pq (0=black … 1=full scale).
            float avg = features[1] > 0.0f ? features[1] : 0.15f;
            result->shadow_strength = fmaxf(0.0f, fminf(0.30f, (0.30f - avg) * 0.8f));
        }
    }

    // ── Highlight bilateral ────────────────────────────────────────────────────
    // Knee tracks Oracle gamma: brighter/flatter curve → lower compression
    // point → knee shifts to match where the curve starts rolling off.
    float safe_gamma = fmaxf(result->gamma, 0.5f);
    result->highlight_knee = fmaxf(0.65f,
                                   fminf(0.85f, 0.90f - (1.0f / safe_gamma) * 0.20f));
    result->highlight_strength = 0.0f;
    if (params->highlight_mode == PL_ML_CONTROL_MANUAL) {
        result->highlight_strength = fmaxf(0.0f, fminf(0.4f, params->highlight_strength));
    } else if (params->highlight_mode == PL_ML_CONTROL_AUTO) {
        if (params->highlight_model) {
            struct pl_ml_prediction hi_pred;
            if (pl_ml_context_predict(params->highlight_model, features, 88, &hi_pred))
                result->highlight_strength = fmaxf(0.0f, fminf(0.4f, hi_pred.value));
        } else {
            // Heuristic: higher peak → more specular clipping risk → more smoothing.
            // features[0] ≈ maxscl/l1max_pq.
            float peak = features[0] > 0.0f ? features[0] : 0.5f;
            result->highlight_strength = fmaxf(0.0f, fminf(0.25f, (peak - 0.55f) * 0.6f));
        }
    }

    return true;
}

int pl_ml_render_get_hooks(struct pl_ml_render_result *result,
                           struct pl_hook *hooks)
{
    if (!result || !hooks)
        return 0;
    int count = 0;
    // SDR→P5 emulation runs before everything else at RGB_INPUT so the shadow
    // mask and every downstream stage see the virtual-PQ (DV-mastered) signal.
    if (result->sdr_emulate) {
        hooks[count++] = (struct pl_hook) {
            .stages = PL_HOOK_RGB_INPUT,
            .input  = PL_HOOK_SIG_TEX,
            .priv   = result,
            .hook   = sdr_p5_hook,
            .signature = 0x5344525035465348ull,  // "SDRP5SH"
        };
    }
    // Shadow bilateral runs FIRST — on the raw PQ input before linearization.
    // (After the emulation bridge when SDR→P5 is active.)
    if (result->shadow_strength > 0.001f) {
        hooks[count++] = (struct pl_hook) {
            .stages = PL_HOOK_RGB_INPUT,
            .input  = PL_HOOK_SIG_TEX,
            .priv   = result,
            .hook   = shadow_bilateral_hook,
            .signature = 0x4D4C53484244574Cull,  // "MLSHBDWL"
        };
    }
    if (result->fire_pop_strength > 0.0f) {
        hooks[count++] = (struct pl_hook) { .stages = PL_HOOK_OUTPUT,
            .input = PL_HOOK_SIG_COLOR, .priv = result, .hook = fire_hook,
            .signature = 0x4456504649524550ull };
    }
    if (result->l2_power != 2048.0f || result->l2_saturation != 2048.0f) {
        hooks[count++] = (struct pl_hook) { .stages = PL_HOOK_OUTPUT,
            .input = PL_HOOK_SIG_COLOR, .priv = result, .hook = l2_hook,
            .signature = 0x44564C325452494Dull };
    }
    if (result->cr_strength > 0.001f) {
        hooks[count++] = (struct pl_hook) {
            .stages = PL_HOOK_OUTPUT,
            .input  = PL_HOOK_SIG_TEX,
            .priv   = result,
            .hook   = cr_bilateral_hook,
            .signature = 0x434F4E5354524352ull,  // "CONSTRCR"
        };
    }
    if (result->radiance.strength > 0.0f)
        pl_ml_radiance_get_hook(&result->radiance, &hooks[count++]);
    if (result->chroma_neutral_boost > 1.0f || result->chroma_fire_boost > 1.0f) {
        hooks[count++] = (struct pl_hook) { .stages = PL_HOOK_OUTPUT,
            .input = PL_HOOK_SIG_COLOR, .priv = result, .hook = chroma_tuner_hook,
            .signature = 0x4348524D41544E55ull };
    }
    // Highlight bilateral runs LAST — after tone-mapping + all color shaping.
    if (result->highlight_strength > 0.001f) {
        hooks[count++] = (struct pl_hook) {
            .stages = PL_HOOK_OUTPUT,
            .input  = PL_HOOK_SIG_TEX,
            .priv   = result,
            .hook   = highlight_bilateral_hook,
            .signature = 0x4D4C48494C545348ull,  // "MLHILTSH"
        };
    }
    return count;
}
