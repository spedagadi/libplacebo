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
#include <libplacebo/ml_features.h>
#include <libplacebo/ml_model.h>
#include <libplacebo/ml_radiance.h>
#include <libplacebo/renderer.h>

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
    float target_nits;
    float l1_max_pq;
    float l1_avg_pq;
    float top_bar_norm;
    float bottom_bar_norm;
    // Optional: persistent GPU cache for feature extraction.
    // Pass a pl_ml_feature_cache created once per session to avoid
    // per-frame luma texture and renderer allocation.
    pl_ml_feature_cache feature_cache;
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
};

// Evaluates canonical frame features and resolves native ML rendering policy.
// The result is valid for configuring one render of `frame`.
PL_API bool pl_ml_render_evaluate(pl_gpu gpu, const struct pl_frame *frame,
                                  const struct pl_ml_render_params *params,
                                  struct pl_ml_render_result *result);

// Initializes output-stage hooks for the resolved L2, fire-pop, radiance, and
// chroma tuner settings. `hooks` must contain room for four entries and remain
// alive for the render using the result.
PL_API int pl_ml_render_get_hooks(struct pl_ml_render_result *result,
                                  struct pl_hook hooks[]);

PL_API_END

#endif
