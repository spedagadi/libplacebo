#include <math.h>

#include <libplacebo/ml_features.h>
#include <libplacebo/ml_render.h>
#include <libplacebo/tone_mapping.h>

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
    static const char body[] =
        "float y = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));\n"
        "float r_res = color.r - y;\n"
        "float b_res = color.b - y;\n"
        "float cb = b_res / 1.8556;\n"
        "float cr = r_res / 1.5748;\n"
        "float dr = (cr - 0.15) / 0.08;\n"
        "float db = (cb + 0.05) / 0.04;\n"
        "float skin_ellipse = dr * dr + db * db;\n"
        "float skin_weight = 1.0 / (1.0 + skin_ellipse * 12.0);\n"
        "float luma_boost = u_chroma_neutral_boost;\n"
        "float denom = 1.0 - u_chroma_knee;\n"
        "if (y > u_chroma_knee && denom > 0.001) {\n"
        "    float t = clamp((y - u_chroma_knee) / denom, 0.0, 1.0);\n"
        "    luma_boost = mix(u_chroma_neutral_boost, u_chroma_fire_boost, pow(t, 1.5));\n"
        "}\n"
        "float chroma_scalar = mix(luma_boost, 1.0, clamp(skin_weight * u_chroma_skin_protect, 0.0, 1.0));\n"
        "chroma_scalar = clamp(chroma_scalar, 1.0, 1.50);\n"
        "// Safe taper: prevent green channel from going negative on saturated highlights\n"
        "// (e.g. pure-red fire pixels where green reconstruction would clip to black)\n"
        "float g_denom = (0.2126 / 0.7152) * r_res + (0.0722 / 0.7152) * b_res;\n"
        "if (g_denom > 0.0) {\n"
        "    float max_safe = y / g_denom;\n"
        "    if (max_safe < chroma_scalar) chroma_scalar = max(1.0, max_safe);\n"
        "}\n"
        "color.r = clamp(y + r_res * chroma_scalar, 0.0, 1.0);\n"
        "color.g = clamp(y - (0.2126 / 0.7152) * r_res * chroma_scalar - (0.0722 / 0.7152) * b_res * chroma_scalar, 0.0, 1.0);\n"
        "color.b = clamp(y + b_res * chroma_scalar, 0.0, 1.0);\n";
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

static bool build_model_features(pl_gpu gpu, const struct pl_frame *frame,
                                 const struct pl_ml_render_params *params,
                                 float features[88])
{
    if (!pl_extract_ml_features(gpu, frame, pl_ml_feature_params(
            .target_nits   = params->target_nits,
            .cache         = params->feature_cache), features))
        return false;

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
    for (int index = 0; index < 8; index++)
        features[77 + index] = spline_lut[knot_indices[index]];
    features[85] = params->top_bar_norm;
    features[86] = params->bottom_bar_norm;
    features[87] = 0.5444f / fmaxf(features[0], 1e-6f);
    return true;
}

bool pl_ml_render_evaluate(pl_gpu gpu, const struct pl_frame *frame,
                           const struct pl_ml_render_params *params,
                           struct pl_ml_render_result *result)
{
    if (!gpu || !frame || !params || !result)
        return false;

    float features[88];
    if (!build_model_features(gpu, frame, params, features))
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
    return true;
}

int pl_ml_render_get_hooks(struct pl_ml_render_result *result,
                           struct pl_hook *hooks)
{
    if (!result || !hooks)
        return 0;
    int count = 0;
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
    return count;
}
