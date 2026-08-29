/*
 * This file is part of libplacebo.
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 */

#ifndef LIBPLACEBO_ML_RADIANCE_H_
#define LIBPLACEBO_ML_RADIANCE_H_

#include <libplacebo/common.h>
#include <libplacebo/shaders/custom.h>

PL_API_BEGIN

enum pl_ml_control_mode {
    PL_ML_CONTROL_OFF,
    PL_ML_CONTROL_AUTO,
    PL_ML_CONTROL_MANUAL,
};

struct pl_ml_radiance_params {
    enum pl_ml_control_mode mode;
    float average_luma;
    float knee;
    float strength;
    float shoulder; // PQ level where boost starts to fade (default 0.82)
                    // prevents specular glare on faces and fire hotspots
};

#define pl_ml_radiance_params(...) (&(struct pl_ml_radiance_params) { __VA_ARGS__ })

struct pl_ml_radiance {
    enum pl_ml_control_mode mode;
    float knee;
    float strength;
    float shoulder; // PQ level where boost fade begins (set by pl_ml_radiance_configure)
};

// Configures adaptive highlight lift. AUTO derives knee and strength from the
// canonical frame average; MANUAL uses the caller values; OFF disables it.
PL_API void pl_ml_radiance_configure(struct pl_ml_radiance *radiance,
                                     const struct pl_ml_radiance_params *params);

// Initializes an output-stage GPU hook backed by `radiance`. The caller keeps
// both structures alive while the hook is registered in pl_render_params.
PL_API void pl_ml_radiance_get_hook(struct pl_ml_radiance *radiance,
                                    struct pl_hook *hook);

PL_API_END

#endif