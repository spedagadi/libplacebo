/*
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LIBPLACEBO_ML_FEATURES_H_
#define LIBPLACEBO_ML_FEATURES_H_

#include <libplacebo/common.h>
#include <libplacebo/gpu.h>
#include <libplacebo/renderer.h>

PL_API_BEGIN

// ML Feature Extraction for Dynamic Tone Mapping
//
// Extracts 77-dimensional feature vector from a frame for ML-based tone mapping.
// This is the SINGLE SOURCE OF TRUTH for feature extraction - both training
// and inference use this identical code to prevent feature skew.
//
// Feature breakdown:
//   [0]       maxscl              - Peak luminance (max PQ)
//   [1]       average_maxrgb      - Average luminance (avg PQ)
//   [2]       fraction_bright     - Fraction of pixels > 0.5 PQ
//   [3-8]     percentiles         - Luma percentiles (p25,p50,p75,p90,p95,p99)
//   [9-17]    zone_mean_3x3       - 3×3 SAT grid mean luma
//   [18-26]   zone_max_3x3        - 3×3 SAT grid max luma
//   [27-51]   zone_mean_5x5       - 5×5 SAT grid mean luma
//   [52-76]   zone_max_5x5        - 5×5 SAT grid max luma
//   [77]      target_nits         - Display target brightness

#define PL_ML_FEATURE_DIM 78

// Persistent GPU resource cache for feature extraction.
// Create once per session, pass to pl_ml_feature_params.cache every frame.
// Eliminates per-frame renderer and texture allocation overhead.
typedef struct pl_ml_feature_cache_t *pl_ml_feature_cache;

PL_API pl_ml_feature_cache pl_ml_feature_cache_create(pl_gpu gpu,
                                                       int width, int height);
PL_API void pl_ml_feature_cache_destroy(pl_ml_feature_cache *cache);

struct pl_ml_feature_params {
    // Target display brightness in nits (e.g., 100, 600, 1000, 4000)
    // This becomes feature[77] and enables multi-display learning
    float target_nits;

    // Downscale resolution for feature extraction
    // Smaller = faster, larger = more accurate spatial features
    // Recommended: 256×144 (good balance)
    int downsample_width;
    int downsample_height;

    // If true, downloads texture to CPU for reference/debug computation.
    // If false, uses GPU histogram and zone-statistics passes when supported.
    bool force_cpu_fallback;

    // Optional: Save downscaled luma for debugging/validation
    // If non-NULL, writes downsample_width * downsample_height float32 values
    // Normalized to [0, 1] range (row-major, height × width)
    const char *debug_luma_path;

    // Optional persistent GPU cache. When provided, luma texture and renderer
    // are reused across frames — eliminates per-frame GPU allocation overhead.
    pl_ml_feature_cache cache;
};

#define PL_ML_FEATURE_DEFAULTS \
    .target_nits = 100.0f, \
    .downsample_width = 256, \
    .downsample_height = 144, \
    .force_cpu_fallback = false, \
    .debug_luma_path = NULL,

#define pl_ml_feature_params(...) (&(struct pl_ml_feature_params) { PL_ML_FEATURE_DEFAULTS __VA_ARGS__ })

// Extract ML features from a frame.
//
// Parameters:
//   gpu       - GPU context for rendering/compute
//   frame     - Input frame (must be in decoded RGB/YUV)
//   params    - Feature extraction parameters
//   features  - Output buffer (must be PL_ML_FEATURE_DIM floats)
//
// Returns true on success, false on error.
//
// Implementation notes:
//   - Frame is downscaled to params->downsample_width × downsample_height
//   - Luma is extracted in PQ space [0, 1]
//   - SAT (Summed Area Table) is computed for spatial zones
//   - All features are bit-exact with inference usage
//
// Performance:
//   - Normal mode reads back only the compact feature statistics.
//   - force_cpu_fallback or debug_luma_path retains the full-luma CPU path.
PL_API bool pl_extract_ml_features(pl_gpu gpu,
                                    const struct pl_frame *frame,
                                    const struct pl_ml_feature_params *params,
                                    float features[PL_ML_FEATURE_DIM]);

// Reset any cached state for feature extraction.
// Call this when switching video sources or on seeks.
PL_API void pl_reset_ml_feature_state(pl_gpu gpu);

PL_API_END

#endif // LIBPLACEBO_ML_FEATURES_H_
