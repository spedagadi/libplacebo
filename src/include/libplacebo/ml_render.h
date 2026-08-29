/*
 * This file is part of libplacebo.
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 */

#ifndef LIBPLACEBO_ML_RENDER_H_
#define LIBPLACEBO_ML_RENDER_H_

#include <libplacebo/gpu.h>
#include <libplacebo/ml_model.h>
#include <libplacebo/ml_radiance.h>
#include <libplacebo/renderer.h>
#include <libplacebo/shaders/colorspace.h>

PL_API_BEGIN

struct pl_ml_render_params {
    pl_ml_context model;
    enum pl_ml_control_mode gamma_mode;
    float gamma;
    enum pl_ml_control_mode cr_mode;
    float cr_strength;
    enum pl_ml_control_mode fire_pop_mode;
    float fire_pop_strength;
    struct pl_ml_radiance_params radiance;
    enum pl_ml_control_mode chroma_mode;
    float chroma_neutral_boost;
    float chroma_fire_boost;
    float chroma_knee;
    float chroma_skin_protect;

    // Shadow bilateral — pre-curve 5×5 bilateral, masks low-PQ zones (y < 0.35).
    // Model predicts sigma_range strength from the shared 88-dim feature vector.
    // If model is NULL and mode is AUTO, uses a scene-luminance heuristic.
    pl_ml_context shadow_model;
    enum pl_ml_control_mode shadow_mode;
    float shadow_strength;   // [0,0.5] manual override

    // Highlight bilateral — post-curve 5×5 bilateral, masks near-clip highlights.
    // Knee is derived from Oracle gamma so it tracks the compressed output range.
    pl_ml_context highlight_model;
    enum pl_ml_control_mode highlight_mode;
    float highlight_strength; // [0,0.4] manual override

    float target_nits;
    float l1_max_pq;
    float l1_avg_pq;
    float top_bar_norm;
    float bottom_bar_norm;
    // The renderer whose peak detection buffer provides ML features.
    // Must be the same pl_renderer used to render the current frame.
    // Features are read via pl_renderer_get_ml_features() — no GPU pass issued.
    pl_renderer renderer;
};

#define pl_ml_render_params(...) (&(struct pl_ml_render_params) { __VA_ARGS__ })

struct pl_ml_render_result {
    bool model_used;
    bool model_fallback;
    float gamma;
    float cr_strength;
    float l2_power;
    float l2_saturation;
    float fire_pop_strength;
    float l1_max_pq;
    float l1_avg_pq;
    struct pl_ml_radiance radiance;
    float chroma_neutral_boost;
    float chroma_fire_boost;
    float chroma_knee;
    float chroma_skin_protect;
    // Shadow/highlight bilateral strengths resolved by pl_ml_render_evaluate.
    float shadow_strength;    // bilateral blend in dark zones [0,0.5]
    float shadow_knee;        // PQ luma boundary for shadow mask (0.35 fixed)
    float highlight_strength; // bilateral blend in bright zones [0,0.4]
    float highlight_knee;     // output-space luma boundary, derived from gamma
};

// Evaluates ML features from the renderer's peak detection state and resolves
// the native ML rendering policy for the current frame.  Features are read
// from params->peak_detect_state — no GPU render pass is issued.
PL_API bool pl_ml_render_evaluate(const struct pl_ml_render_params *params,
                                  struct pl_ml_render_result *result);

// Initializes hooks for the resolved ML settings. `hooks` must contain room
// for six entries and remain alive for the render using the result.
// Slot order: [0] shadow bilateral (RGB_INPUT), [1] highlight bilateral
// (OUTPUT), [2] fire-pop, [3] L2 gamma, [4] radiance, [5] chroma tuner.
// Returns the actual number of hooks populated (0–6).
PL_API int pl_ml_render_get_hooks(struct pl_ml_render_result *result,
                                  struct pl_hook hooks[]);

PL_API_END

#endif
