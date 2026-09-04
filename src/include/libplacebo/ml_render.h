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
    // Trained P(skin) chroma LUT (r8, bilinear + clamp). When non-NULL and
    // skin_lut_gpu matches the rendering GPU, chroma_tuner_hook samples the LUT
    // instead of the hand-tuned ellipse and skips the highlight taper. Caller
    // owns the texture lifecycle; pl_ml_render_evaluate forwards to the result
    // on every frame (all controls modes). Bounds are the LUT grid edges in
    // the canonical encoded (cr, cb) space.
    pl_tex skin_lut;
    pl_gpu skin_lut_gpu;
    float skin_lut_cr0, skin_lut_cr1;   // cr grid bounds
    float skin_lut_cb0, skin_lut_cb1;   // cb grid bounds
    // Post-tone-mapping hue correction (degrees). The IPT working space used
    // by libplacebo's tone mapper introduces a systematic hue rotation toward
    // magenta; this counter-rotates Cr/Cb in the L2 output hook. Negative =
    // less magenta. Typical value: -7.
    float hue_correction;

    // Hunt effect compensation: boosts chroma to counteract the perceived
    // colorfulness loss from HDR→SDR luminance compression (CAM16-derived).
    // Scales with l1_max_pq² so bright HDR gets more, near-SDR gets none.
    // 0.0 = off, 0.80 = default, 1.0 = aggressive.
    float hunt_compensation;
    // Highlight desaturation ceiling: max chroma reduction in the highlight
    // guard zone (smoothstep from l2_guard to 0.95). 0.0 = off, 0.12 = default,
    // 0.50 = aggressive (old default).
    float highlight_desat;

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

    float headroom;          // [0,0.30] highlight headroom override (-1 = auto)
    float hdr_detail;        // [0,1] HDR texture recovery injection strength
    float vector_desat;      // [0,0.6] pre-TM highlight desaturation

    // Two-stage tone mapping: when virtual_target_nits > 0 and exceeds the
    // real display peak, libplacebo tone-maps to the virtual target (gentler
    // curve, more highlight texture preserved). A Zion Core OUTPUT hook then
    // compresses the wider-range signal to the real display peak using a
    // soft-knee Reinhard curve. Set virtual_target_nits=0 to disable.
    float virtual_target_nits;  // [0, 400] virtual TM target (0 = off)

    float target_nits;
    float l1_max_pq;
    float l1_avg_pq;
    float top_bar_norm;
    float bottom_bar_norm;
    // SDR source (transfer not PQ/HLG): ML features are computed in true
    // PQ-of-nits (bt.1886 → nits → PQ), and the 77–84 spline knots are
    // synthesized from a virtual DV-mastered L2 curve (1000-nit ceiling).
    bool is_sdr;
    // SDR→P5 virtual-master bridge: when is_sdr AND emulate_sdr, a RGB_INPUT
    // hook decodes SDR → linear nits → scales to the virtual P5 ceiling
    // (sdr_virtual_nits, default 1000) → re-encodes as PQ and re-labels the
    // frame PQ/sig_peak=virtual, so tone-mapping, feature detection and the
    // ML grade all run on a virtual HDR master. When active, the raw-SDR
    // feature branches (synthetic pivots, true-PQ shader path) are bypassed.
    bool emulate_sdr;
    float sdr_virtual_nits;   // [100, 4000] virtual P5 ceiling in nits
    float sdr_strength;       // [0, 1] shaped mid-lift (0 = pure linear gain,
                              // 1 = strong SDR→HDR "lift" curve that survives
                              // tone-mapping)
    // Dehaze (shadow crush): steepens the near-black ramp so haze from
    // tone-mapping compression is pushed toward true black. -1 = auto
    // (scene-adaptive), 0 = off, 0..1 = manual strength.
    float dehaze;

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
    float l2_highlight_guard; // luma where highlight protection starts (adaptive)
    float hunt_strength;      // Hunt effect chroma boost coefficient
    float highlight_desat;    // highlight desaturation ceiling [0,0.50]
    float fire_pop_strength;
    float l1_max_pq;
    float l1_avg_pq;
    struct pl_ml_radiance radiance;
    float chroma_neutral_boost;
    float chroma_fire_boost;
    float chroma_knee;
    float chroma_skin_protect;
    // Forwarded from params by pl_ml_render_evaluate (unconditional, all modes),
    // so chroma_tuner_hook can bind the LUT texture. Zero when no LUT is set,
    // in which case the hook uses the legacy hand-tuned ellipse path.
    pl_tex skin_lut;
    pl_gpu skin_lut_gpu;
    float skin_lut_cr0, skin_lut_cr1;
    float skin_lut_cb0, skin_lut_cb1;
    float hue_correction;     // post-TM hue rotation (degrees, forwarded)
    // Shadow/highlight bilateral strengths resolved by pl_ml_render_evaluate.
    float shadow_toe;         // global toe-lift in dark zones [0,0.5]
    float shadow_strength;    // bilateral detail injection in dark zones [0,0.5]
    float shadow_knee;        // display-referred luma boundary (0.25 fixed)
    float highlight_rolloff;  // shoulder adjustment [-0.30,+0.40], signed
    float highlight_strength; // bilateral smoothing in bright zones [0,0.4]
    float highlight_knee;     // output-space luma boundary, derived from gamma
    float headroom;           // highlight headroom [0,0.20] — soft shoulder
                              // compression before ML hooks to prevent clipping
    // HDR texture recovery: captures high-frequency detail from the PQ input
    // and re-injects it into the SDR output to preserve highlight structure
    // (clouds, sand, specular) that tone mapping flattens.
    float hdr_detail_strength; // pre-TM highlight texture enhance [0,1], 0 = off
    float vector_desat_strength; // pre-TM highlight desaturation [0,0.6]
    // Two-stage tone mapping: PQ ceiling for pre-compression.
    // When > 0, the pq_precompress_hook soft-clips PQ values to this
    // ceiling so the tone mapper sees a lower effective content peak.
    float dc_ceiling_pq;    // PQ-encoded ceiling (0 = off)
    float dc_compress_gain; // virtual_nits / real_nits (>1 = compress needed)
    // Dehaze: shadow crush that steepens the near-black ramp, pushing hazy
    // shadows toward true black.  Tunable strength; knee sets the luma
    // boundary below which the crush applies.
    float dehaze_strength;  // [0,1] — 0 = off, higher = deeper blacks
    float dehaze_knee;      // display-referred luma boundary [0.05,0.25]
    // SDR→P5 emulation bridge (resolved): active when is_sdr && emulate_sdr.
    bool sdr_emulate;
    float sdr_virtual_nits;
    float sdr_strength;
};

// Evaluates ML features from the renderer's peak detection state and resolves
// the native ML rendering policy for the current frame.  Features are read
// from params->peak_detect_state — no GPU render pass is issued.
PL_API bool pl_ml_render_evaluate(const struct pl_ml_render_params *params,
                                  struct pl_ml_render_result *result);

// Initializes hooks for the resolved ML settings. `hooks` must contain room
// for thirteen entries and remain alive for the render using the result.
// Slot order: [0] SDR→P5 emulation (NATIVE), [1] PQ pre-compress
// (NATIVE), [2] vector desaturation (NATIVE), [3] HDR detail
// enhance (NATIVE), [4] display compress (OUTPUT, first — stage 2 of
// two-stage TM), [5] headroom shoulder, [6] shadow toe-lift + bilateral,
// [7] highlight roll-off + bilateral, [8] fire-pop, [9] L2 gamma,
// [10] CR bilateral, [11] radiance, [12] chroma tuner.
// Returns the actual number of hooks populated (0–13).
PL_API int pl_ml_render_get_hooks(struct pl_ml_render_result *result,
                                  struct pl_hook hooks[]);

PL_API_END

#endif
