#include <math.h>

#include <libplacebo/ml_radiance.h>

static struct pl_hook_res radiance_hook(void *priv,
                                        const struct pl_hook_params *params)
{
    struct pl_ml_radiance *radiance = priv;
    pl_shader sh = params->sh;
    // Resonance: bell-shaped highlight lift.
    //
    // Rising side (knee → shoulder): smooth-step envelope lifts mid-highlights
    // toward white, recovering HDR luminous depth.
    //
    // Soft shoulder (shoulder → 1.0): envelope decays back toward zero.
    // Without this, concentrated bright regions (fire hotspot, specular glare
    // on faces) receive full boost → visible bloom / over-brightening.
    // The shoulder fade creates a gentle rolloff so the boost peaks in the
    // mid-highlight zone and retreats before reaching specular/fire whites.
    static const char body[] =
        "float y = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));\n"
        // Rising envelope above knee
        "float t = clamp((y - radiance_knee) / (1.0 - radiance_knee), 0.0, 1.0);\n"
        "float envelope = t * t * (3.0 - 2.0 * t);\n"
        // Soft shoulder: fade envelope back to 0 above shoulder threshold
        "float s = clamp((y - radiance_shoulder) / (1.0 - radiance_shoulder), 0.0, 1.0);\n"
        "float shoulder_fade = 1.0 - s * s * (3.0 - 2.0 * s);\n"
        "float eff_env = envelope * shoulder_fade;\n"
        // Apply boost with shoulder-limited envelope
        "float target_y = mix(y, 1.0, eff_env);\n"
        "float final_y = mix(y, target_y, radiance_strength);\n"
        "float chroma_taper = 1.0 - 0.14 * eff_env * radiance_strength;\n"
        "float cb = (color.b - y) * chroma_taper;\n"
        "float cr = (color.r - y) * chroma_taper;\n"
        "color.r = clamp(final_y + cr, 0.0, 1.0);\n"
        "color.g = clamp(final_y - 0.2126 / 0.7152 * cr - 0.0722 / 0.7152 * cb, 0.0, 1.0);\n"
        "color.b = clamp(final_y + cb, 0.0, 1.0);\n";
    struct pl_shader_var vars[] = {
        { .var = pl_var_float("radiance_knee"),     .data = &radiance->knee,     .dynamic = true },
        { .var = pl_var_float("radiance_strength"), .data = &radiance->strength, .dynamic = true },
        { .var = pl_var_float("radiance_shoulder"), .data = &radiance->shoulder, .dynamic = true },
    };
    if (!pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "GPU adaptive radiance highlight lift",
        .body = body,
        .input = PL_SHADER_SIG_COLOR,
        .output = PL_SHADER_SIG_COLOR,
        .variables = vars,
        .num_variables = 3,
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

void pl_ml_radiance_configure(struct pl_ml_radiance *radiance,
                              const struct pl_ml_radiance_params *params)
{
    if (!radiance || !params)
        return;
    radiance->mode = params->mode;
    radiance->knee = 0.60f;
    radiance->strength = 0.0f;
    radiance->shoulder = 0.82f; // default: soft rolloff above 0.82 PQ
    if (params->mode == PL_ML_CONTROL_AUTO) {
        float avg = fmaxf(0.05f, fminf(0.40f, params->average_luma));
        radiance->knee = 0.40f + avg;
        radiance->strength = fmaxf(0.15f, 0.60f * (1.0f - 1.5f * avg));
        // Auto shoulder: set above the auto knee with some headroom
        radiance->shoulder = fminf(0.95f, radiance->knee + 0.25f);
    } else if (params->mode == PL_ML_CONTROL_MANUAL) {
        radiance->knee = fmaxf(0.40f, fminf(0.90f, params->knee));
        radiance->strength = fmaxf(0.0f, fminf(0.60f, params->strength));
        radiance->shoulder = fmaxf(radiance->knee + 0.05f,
                                   fminf(0.99f, params->shoulder));
    }
}

void pl_ml_radiance_get_hook(struct pl_ml_radiance *radiance,
                             struct pl_hook *hook)
{
    if (!hook)
        return;
    *hook = (struct pl_hook) {
        .stages = PL_HOOK_OUTPUT,
        .input = PL_HOOK_SIG_COLOR,
        .priv = radiance,
        .hook = radiance_hook,
        .signature = 0x445652414449414Eull,
    };
}
