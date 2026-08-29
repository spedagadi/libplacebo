#include <math.h>

#include <libplacebo/dispatch.h>
#include <libplacebo/ml_render.h>
#include <libplacebo/tone_mapping.h>
#include <libplacebo/shaders/colorspace.h>

static struct pl_hook_res l2_hook(void *priv, const struct pl_hook_params *params)
{
    struct pl_ml_render_result *result = priv;
    pl_shader sh = params->sh;
    static const char body[] =
        "float y = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));\n"
        "float cb = (color.b - y) * (l2_saturation / 2048.0);\n"
        "float cr = (color.r - y) * (l2_saturation / 2048.0);\n"
        "float l2_gamma = 2048.0 / l2_power;\n"
        "if (l2_gamma != 1.0 && y > 0.001 && y < 0.999) {\n"
        "    float t = 2.0 * y - 1.0;\n"
        "    y = 0.5 * (sign(t) * pow(abs(t), l2_gamma) + 1.0);\n"
        "}\n"
        "color.r = clamp(y + cr, 0.0, 1.0);\n"
        "color.g = clamp(y - 0.2126 / 0.7152 * cr - 0.0722 / 0.7152 * cb, 0.0, 1.0);\n"
        "color.b = clamp(y + cb, 0.0, 1.0);\n";
    struct pl_shader_var vars[] = {
        { .var = pl_var_float("l2_power"), .data = &result->l2_power, .dynamic = true },
        { .var = pl_var_float("l2_saturation"), .data = &result->l2_saturation, .dynamic = true },
    };
    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "GPU L2 gamma and saturation trim", .body = body,
        .input = PL_SHADER_SIG_COLOR, .output = PL_SHADER_SIG_COLOR,
        .variables = vars, .num_variables = 2,
        .output_w = pl_rect_w(params->dst_rect), .output_h = pl_rect_h(params->dst_rect),
    })) return (struct pl_hook_res) { .failed = true };
    return (struct pl_hook_res) { .output = PL_HOOK_SIG_COLOR, .sh = sh,
        .repr = params->repr, .color = params->color, .components = params->components,
        .rect = params->rect };
}

static struct pl_hook_res fire_hook(void *priv, const struct pl_hook_params *params)
{
    struct pl_ml_render_result *result = priv;
    pl_shader sh = params->sh;
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
    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "GPU fire-pop output hook", .body = body,
        .input = PL_SHADER_SIG_COLOR, .output = PL_SHADER_SIG_COLOR,
        .variables = &var, .num_variables = 1,
        .output_w = pl_rect_w(params->dst_rect), .output_h = pl_rect_h(params->dst_rect),
    })) return (struct pl_hook_res) { .failed = true };
    return (struct pl_hook_res) { .output = PL_HOOK_SIG_COLOR, .sh = sh,
        .repr = params->repr, .color = params->color, .components = params->components,
        .rect = params->rect };
}

static struct pl_hook_res chroma_tuner_hook(void *priv, const struct pl_hook_params *params)
{
    struct pl_ml_render_result *result = priv;
    pl_shader sh = params->sh;
    // Perceptual-space chroma tuner.
    //
    // Root cause of facial blowout: PL_HOOK_OUTPUT runs in LINEAR light.
    // YCbCr skin-ellipse parameters were calibrated for gamma-encoded space.
    // In linear light, warm highlights cause r_channel to spike exponentially,
    // inflating cr far outside the ellipse → skin pixels flagged as fire →
    // full fire_boost applied → orange blowout on faces.
    //
    // Fix: convert to perceptual (gamma 2.2) space before skin detection,
    // apply the vector boost there, then invert back to linear.
    // This restores the cr/cb ratios to the range the ellipse expects.
    static const char body[] =
        // 1. Convert linear → perceptual for accurate skin math
        "vec3 p = pow(clamp(color.rgb, 0.0, 1.0), vec3(1.0 / 2.2));\n"
        "float y = dot(p, vec3(0.2126, 0.7152, 0.0722));\n"
        "float r_res = p.r - y;\n"
        "float b_res = p.b - y;\n"
        "float cb = b_res / 1.8556;\n"
        "float cr = r_res / 1.5748;\n"
        // 2. Adaptive ellipse expansion above knee (highlight coordinate drift fix)
        "float dr = (cr - 0.15) / 0.08;\n"
        "float db = (cb + 0.05) / 0.04;\n"
        "if (y > u_chroma_knee) {\n"
        "    float hd = clamp((y - u_chroma_knee) / (1.0 - u_chroma_knee), 0.0, 1.0);\n"
        "    float ex = 1.0 + hd * 0.50;\n"
        "    dr /= ex; db /= ex;\n"
        "}\n"
        "float skin_ellipse = dr * dr + db * db;\n"
        "float skin_weight = 1.0 / (1.0 + skin_ellipse * 12.0);\n"
        // 3. Adaptive highlight skin taper (perceptual space).
        // Starts at y=0.60 perceptual — covers Boromir-style specular blind spot
        // (0.60–0.74) while leaving normal skin midtones (y < 0.60) free to
        // receive the full color boost.
        // NOTE: taper_start must be a fixed perceptual value, NOT derived from
        // u_chroma_knee. The knee (0.55) was calibrated in linear space; using
        // knee-0.05=0.50 in perceptual maps to ~22% linear and over-protects
        // 80%+ of the image, killing neutral_boost entirely.
        "if (y > 0.60) {\n"
        "    float dt = clamp((y - 0.60) / 0.40, 0.0, 1.0);\n"  // ramp 0.60→1.0
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
        // 6. Apply scalar to perceptual residuals, reconstruct, invert to linear
        "r_res *= chroma_scalar; b_res *= chroma_scalar;\n"
        "vec3 perc_out;\n"
        "perc_out.r = y + r_res;\n"
        "perc_out.g = y - (0.2126 / 0.7152) * r_res - (0.0722 / 0.7152) * b_res;\n"
        "perc_out.b = y + b_res;\n"
        "color.rgb = pow(clamp(perc_out, 0.0, 1.0), vec3(2.2));\n";
    struct pl_shader_var vars[] = {
        { .var = pl_var_float("u_chroma_neutral_boost"), .data = &result->chroma_neutral_boost, .dynamic = true },
        { .var = pl_var_float("u_chroma_fire_boost"),    .data = &result->chroma_fire_boost,    .dynamic = true },
        { .var = pl_var_float("u_chroma_knee"),          .data = &result->chroma_knee,          .dynamic = true },
        { .var = pl_var_float("u_chroma_skin_protect"),  .data = &result->chroma_skin_protect,  .dynamic = true },
    };
    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "Adaptive Chroma Vector Tuner", .body = body,
        .input = PL_SHADER_SIG_COLOR, .output = PL_SHADER_SIG_COLOR,
        .variables = vars, .num_variables = 4,
    })) return (struct pl_hook_res) { .failed = true };
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

    struct pl_shader_var vars[] = {
        { .var = pl_var_float("shadow_str"),  .data = &result->shadow_strength, .dynamic = true },
        { .var = pl_var_float("shadow_knee"), .data = &result->shadow_knee,     .dynamic = true },
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
        "float y_c = dot(center, vec3(0.2126, 0.7152, 0.0722));\n"
        "float smask = 1.0 - smoothstep(shadow_knee * 0.5, shadow_knee, y_c);\n"
        "vec3 out_rgb = center;\n"
        "if (smask > 0.001 && shadow_str > 0.001) {\n"
        "    // 1. 5x5 Bilateral edge-preserving spatial denoiser\n"
        "    vec3 acc = vec3(0.0); float tw = 0.0;\n"
        "    float sr = clamp(shadow_str, 0.02, 0.30);\n"
        "    for (int dy = -2; dy <= 2; dy++) {\n"
        "        for (int dx = -2; dx <= 2; dx++) {\n"
        "            vec3 s = textureLod(shadow_src, uv + vec2(dx, dy) * pt, 0.0).rgb;\n"
        "            float sy = dot(s, vec3(0.2126, 0.7152, 0.0722));\n"
        "            float d2 = float(dx*dx + dy*dy);\n"
        "            float yd = y_c - sy;\n"
        "            float w = exp(-d2 * 0.222) * exp(-yd*yd / (2.0*sr*sr));\n"
        "            acc += w * s; tw += w;\n"
        "        }\n"
        "    }\n"
        "    vec3 denoised = acc / max(tw, 0.001);\n"
        "    out_rgb = mix(center, denoised, smask * clamp(shadow_str * 2.0, 0.0, 1.0));\n"
        "    // 2. Machine-learned shadow toe (un-crush dark gradients)\n"
        "    // Multiplicative gain anchored at black: out = rgb*(1+str*(1-u)^2)\n"
        "    // u = y_c/knee. At u=0 the gain is 1+str (max detail expansion, no\n"
        "    // lift of black); at u=1 the gain is 1 and the curve merges C1.\n"
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
    // libplacebo tone-map spline — same as before, CPU-only.
    struct pl_tone_map_params spline = {
        .function = &pl_tone_map_spline,
        .constants = { PL_TONE_MAP_CONSTANTS },
        .input_scaling = PL_HDR_PQ,
        .output_scaling = PL_HDR_PQ,
        .lut_size = 256,
        .input_max = fmaxf(params->l1_max_pq, features[0]),
        .input_avg = params->l1_avg_pq > 0.0f ? params->l1_avg_pq : features[1],
        .output_max = 0.5444f,
    };
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
    features[87] = 0.5444f / fmaxf(features[0], 1e-6f);
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

    if (params->cr_mode == PL_ML_CONTROL_MANUAL) {
        result->cr_strength = params->cr_strength;
    } else if (params->cr_mode == PL_ML_CONTROL_AUTO) {
        float base_cr = fmaxf(0.1f, fminf(0.5f,
            0.25f + (1.2f - result->gamma) * 0.15f));
        // Taper CR for very bright/high-contrast scenes (fires, explosions).
        // At l1_max > 0.7 the bilateral filter has extreme gradients that crush
        // bright edges. Scale back up to 50% reduction at l1_max=1.0.
        float l1max = result->l1_max_pq > 0.0f ? result->l1_max_pq :
                      (params->l1_max_pq > 0.0f ? params->l1_max_pq : 0.5f);
        float brightness_taper = 1.0f - fmaxf(0.0f, (l1max - 0.7f) / 0.3f) * 0.5f;
        result->cr_strength = base_cr * brightness_taper;
    }

    struct pl_ml_radiance_params radiance = params->radiance;
    radiance.average_luma = features[1];
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
                result->highlight_strength = fmaxf(0.0f, fminf(0.4f, hi_pred.gamma));
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
    // Shadow bilateral runs FIRST — on the raw PQ input before linearization.
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
