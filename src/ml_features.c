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

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "common.h"
#include "log.h"
#include "gpu.h"
#include "shaders.h"

#include <libplacebo/ml_features.h>
#include <libplacebo/renderer.h>
#include <libplacebo/colorspace.h>
#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/sampling.h>
#include <libplacebo/utils/libav.h>

// Comparison function for percentile sorting
static int cmp_float(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

static const char zone_stats_shader[] =
    "#version 450\n"
    "layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
    "layout (binding = 0) uniform sampler2D luma;\n"
    "layout (rgba32f, binding = 1) writeonly uniform image2D stats;\n"
    "void main() {\n"
    "    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);\n"
    "    ivec2 size = textureSize(luma, 0);\n"
    "    int grid = gid.x < 3 ? 3 : 5;\n"
    "    int col = gid.x < 3 ? gid.x : gid.x - 3;\n"
    "    if (gid.y >= grid || col >= grid) return;\n"
    "    int x0 = col * size.x / grid, x1 = (col + 1) * size.x / grid;\n"
    "    int y0 = gid.y * size.y / grid, y1 = (gid.y + 1) * size.y / grid;\n"
    "    float total = 0.0, peak = 0.0;\n"
    "    for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) {\n"
    "        float value = texelFetch(luma, ivec2(x, y), 0).r;\n"
    "        total += value; peak = max(peak, value);\n"
    "    }\n"
    "    imageStore(stats, gid, vec4(total, peak, float((x1-x0)*(y1-y0)), 0.0));\n"
    "}\n";

static bool extract_zone_stats_gpu(pl_gpu gpu, pl_tex luma_tex,
                                   float zone_stats[8][5][3])
{
    pl_fmt stats_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 32,
                                   PL_FMT_CAP_STORABLE | PL_FMT_CAP_HOST_READABLE);
    if (!stats_fmt) return false;
    pl_tex stats_tex = pl_tex_create(gpu, pl_tex_params(
        .w = 8, .h = 5, .format = stats_fmt,
        .storable = true, .host_readable = true));
    if (!stats_tex) return false;

    struct pl_desc descs[] = {
        { .name = "luma", .type = PL_DESC_SAMPLED_TEX, .binding = 0,
          .access = PL_DESC_ACCESS_READONLY },
        { .name = "stats", .type = PL_DESC_STORAGE_IMG, .binding = 1,
          .access = PL_DESC_ACCESS_WRITEONLY },
    };
    pl_pass pass = pl_pass_create(gpu, pl_pass_params(
        .type = PL_PASS_COMPUTE, .descriptors = descs, .num_descriptors = 2,
        .glsl_shader = zone_stats_shader));
    if (!pass) { pl_tex_destroy(gpu, &stats_tex); return false; }

    struct pl_desc_binding bindings[] = {
        { .object = luma_tex }, { .object = stats_tex },
    };
    pl_pass_run(gpu, pl_pass_run_params(
        .pass = pass, .desc_bindings = bindings,
        .compute_groups = { 8, 5, 1 }));

    float raw[8 * 5 * 4] = {0};
    bool ok = pl_tex_download(gpu, pl_tex_transfer_params(
        .tex = stats_tex, .ptr = raw,
        .row_pitch = 8 * 4 * sizeof(float)));
    if (ok) {
        memset(zone_stats, 0, sizeof(float) * 8 * 5 * 3);
        for (int y = 0; y < 5; y++) for (int x = 0; x < 8; x++) {
            int dst = (y * 8 + x) * 4;
            zone_stats[x][y][0] = raw[dst + 0];
            zone_stats[x][y][1] = raw[dst + 1];
            zone_stats[x][y][2] = raw[dst + 2];
        }
    }
    pl_pass_destroy(gpu, &pass);
    pl_tex_destroy(gpu, &stats_tex);
    return ok;
}

static const char histogram_shader[] =
    "#version 450\n"
    "layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;\n"
    "layout (binding = 0) uniform sampler2D luma;\n"
    "layout (std430, binding = 1) buffer histogram { uint bins[]; };\n"
    "void main() {\n"
    "    ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
    "    ivec2 size = textureSize(luma, 0);\n"
    "    if (p.x >= size.x || p.y >= size.y) return;\n"
    "    float value = texelFetch(luma, p, 0).r;\n"
    "    uint bin = uint(clamp(round(value * 65535.0), 0.0, 65535.0));\n"
    "    atomicAdd(bins[bin], 1u);\n"
    "}\n";

static bool reduce_histogram_gpu(pl_gpu gpu, pl_buf hist_buf, float values[9]);

static bool extract_histogram_gpu(pl_gpu gpu, pl_tex luma_tex,
                                  uint32_t histogram[65536], float reduced[9])
{
    const size_t size = 65536 * sizeof(uint32_t);
    uint32_t *zeroes = pl_calloc(NULL, 65536, sizeof(uint32_t));
    if (!zeroes) return false;
    pl_buf hist_buf = pl_buf_create(gpu, pl_buf_params(
        .size = size, .storable = true, .host_readable = true,
        .initial_data = zeroes));
    pl_free(zeroes);
    if (!hist_buf) return false;

    struct pl_desc descs[] = {
        { .name = "luma", .type = PL_DESC_SAMPLED_TEX, .binding = 0,
          .access = PL_DESC_ACCESS_READONLY },
        { .name = "histogram", .type = PL_DESC_BUF_STORAGE, .binding = 1,
          .access = PL_DESC_ACCESS_READWRITE },
    };
    pl_pass pass = pl_pass_create(gpu, pl_pass_params(
        .type = PL_PASS_COMPUTE, .descriptors = descs, .num_descriptors = 2,
        .glsl_shader = histogram_shader));
    if (!pass) { pl_buf_destroy(gpu, &hist_buf); return false; }

    struct pl_desc_binding bindings[] = {
        { .object = luma_tex }, { .object = hist_buf },
    };
    pl_pass_run(gpu, pl_pass_run_params(
        .pass = pass, .desc_bindings = bindings,
        .compute_groups = { 16, 9, 1 }));
    bool ok;
    if (reduced) {
        ok = reduce_histogram_gpu(gpu, hist_buf, reduced);
    } else {
        ok = pl_buf_read(gpu, hist_buf, 0, histogram, size);
    }
    pl_pass_destroy(gpu, &pass);
    pl_buf_destroy(gpu, &hist_buf);
    return ok;
}

static float histogram_percentile(const uint32_t histogram[65536],
                                  int count, float target)
{
    float position = target * (count - 1);
    int lower = (int) position;
    int upper = lower + 1;
    int seen = 0, lower_bin = 0, upper_bin = 0;
    for (int bin = 0; bin < 65536; bin++) {
        int next = seen + (int) histogram[bin];
        if (lower < next) lower_bin = bin;
        if (upper < next) { upper_bin = bin; break; }
        seen = next;
    }
    float frac = position - lower;
    return (lower_bin * (1.0f - frac) + upper_bin * frac) / 65535.0f;
}

static const char histogram_reduce_shader[] =
    "#version 450\n"
    "layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
    "layout (std430, binding = 0) readonly buffer histogram { uint bins[]; };\n"
    "layout (std430, binding = 1) writeonly buffer result { float values[]; };\n"
    "void main() {\n"
    "    uint count = 0u; uint bright = 0u; uint sum = 0u;\n"
    "    uint max_bin = 0u;\n"
    "    for (uint bin = 0u; bin < 65536u; bin++) {\n"
    "        uint n = bins[bin]; count += n;\n"
    "        bright += bin > 32767u ? n : 0u;\n"
    "        sum += bin * n;\n"
    "        if (n > 0u) max_bin = bin;\n"
    "    }\n"
    "    values[0] = float(max_bin) / 65535.0;\n"
    "    values[1] = float(sum) / (65535.0 * float(count));\n"
    "    values[2] = float(bright) / float(count);\n"
    "    const float targets[6] = float[6](0.25, 0.50, 0.75, 0.90, 0.95, 0.99);\n"
    "    for (int p = 0; p < 6; p++) {\n"
    "        uint want = uint(targets[p] * float(count - 1));\n"
    "        uint seen = 0u; uint value = 0u;\n"
    "        for (uint bin = 0u; bin < 65536u; bin++) {\n"
    "            uint next = seen + bins[bin];\n"
    "            if (want < next) { value = bin; break; }\n"
    "            seen = next;\n"
    "        }\n"
    "        values[3 + p] = float(value) / 65535.0;\n"
    "    }\n"
    "}\n";

static bool reduce_histogram_gpu(pl_gpu gpu, pl_buf hist_buf, float values[9])
{
    pl_buf result_buf = pl_buf_create(gpu, pl_buf_params(
        .size = 9 * sizeof(float), .storable = true, .host_readable = true));
    if (!result_buf) return false;
    struct pl_desc descs[] = {
        { .name = "histogram", .type = PL_DESC_BUF_STORAGE, .binding = 0,
          .access = PL_DESC_ACCESS_READONLY },
        { .name = "result", .type = PL_DESC_BUF_STORAGE, .binding = 1,
          .access = PL_DESC_ACCESS_WRITEONLY },
    };
    pl_pass pass = pl_pass_create(gpu, pl_pass_params(
        .type = PL_PASS_COMPUTE, .descriptors = descs, .num_descriptors = 2,
        .glsl_shader = histogram_reduce_shader));
    if (!pass) { pl_buf_destroy(gpu, &result_buf); return false; }
    struct pl_desc_binding bindings[] = {
        { .object = hist_buf }, { .object = result_buf },
    };
    pl_pass_run(gpu, pl_pass_run_params(
        .pass = pass, .desc_bindings = bindings,
        .compute_groups = { 1, 1, 1 }));
    bool ok = pl_buf_read(gpu, result_buf, 0, values, 9 * sizeof(float));
    pl_pass_destroy(gpu, &pass);
    pl_buf_destroy(gpu, &result_buf);
    return ok;
}

static bool extract_features_gpu(pl_gpu gpu, pl_tex luma_tex,
                                 float features[PL_ML_FEATURE_DIM])
{
    float reduced[9] = {0};
    float zones[8][5][3] = {0};
    if (!extract_histogram_gpu(gpu, luma_tex, NULL, reduced) ||
        !extract_zone_stats_gpu(gpu, luma_tex, zones))
        return false;

    for (int i = 0; i < 9; i++) features[i] = reduced[i];

    for (int row = 0; row < 3; row++) for (int col = 0; col < 3; col++) {
        features[9 + row * 3 + col] = zones[col][row][0] / zones[col][row][2];
        features[18 + row * 3 + col] = zones[col][row][1];
    }
    for (int row = 0; row < 5; row++) for (int col = 0; col < 5; col++) {
        features[27 + row * 5 + col] = zones[col + 3][row][0] / zones[col + 3][row][2];
        features[52 + row * 5 + col] = zones[col + 3][row][1];
    }
    return true;
}

// CPU fallback: Extract percentiles and SAT zones from downscaled luma texture
static bool extract_features_cpu(pl_gpu gpu, pl_tex luma_tex,
                                  const struct pl_ml_feature_params *params,
                                  float features[PL_ML_FEATURE_DIM])
{
    PL_INFO(gpu, "[ML-CPU] === extract_features_cpu START ===");
    int w = params->downsample_width;
    int h = params->downsample_height;
    int n_pixels = w * h;
    PL_INFO(gpu, "[ML-CPU] Resolution: %dx%d = %d pixels", w, h, n_pixels);

    // Download texture to CPU (as uint16_t, since format is r16)
    size_t buffer_size = n_pixels * sizeof(uint16_t);
    PL_INFO(gpu, "[ML-CPU] Allocating u16 buffer: %zu bytes", buffer_size);
    uint16_t *luma_u16 = pl_alloc(NULL, buffer_size);
    if (!luma_u16) {
        PL_ERR(gpu, "Failed to allocate CPU buffer for feature extraction");
        return false;
    }
    PL_INFO(gpu, "[ML-CPU] u16 buffer allocated");

    PL_INFO(gpu, "[ML-CPU] Downloading texture to CPU...");
    bool ok = pl_tex_download(gpu, pl_tex_transfer_params(
        .tex = luma_tex,
        .ptr = luma_u16,
    ));

    if (!ok) {
        PL_ERR(gpu, "[ML-CPU] Failed to download luma texture");
        pl_free(luma_u16);
        return false;
    }
    PL_INFO(gpu, "[ML-CPU] Texture downloaded successfully");

    // Convert uint16_t to float [0, 1]
    PL_INFO(gpu, "[ML-CPU] Converting u16 to float...");
    float *luma = pl_alloc(NULL, n_pixels * sizeof(float));
    if (!luma) {
        PL_ERR(gpu, "Failed to allocate float buffer");
        pl_free(luma_u16);
        return false;
    }

    for (int i = 0; i < n_pixels; i++) {
        luma[i] = luma_u16[i] / 65535.0f;
    }
    pl_free(luma_u16);
    PL_INFO(gpu, "[ML-CPU] Conversion complete, u16 buffer freed");

    uint32_t histogram[65536] = {0};
    bool gpu_histogram = !params->force_cpu_fallback &&
                         extract_histogram_gpu(gpu, luma_tex, histogram, NULL);
    PL_WARN(gpu, "[ML] GPU histogram: %s", gpu_histogram ? "enabled" : "fallback");

    // Debug: Save luma for validation if requested
    if (params->debug_luma_path) {
        FILE *f = fopen(params->debug_luma_path, "wb");
        if (f) {
            fwrite(luma, sizeof(float), n_pixels, f);
            fclose(f);
            PL_INFO(gpu, "[ML-CPU] Saved debug luma to: %s", params->debug_luma_path);
        } else {
            PL_WARN(gpu, "[ML-CPU] Failed to save debug luma: %s", params->debug_luma_path);
        }
    }

    // Feature 0: maxscl (peak luminance)
    PL_INFO(gpu, "[ML-CPU] Computing maxscl (feature 0)...");
    float max_luma = 0.0f;
    for (int i = 0; i < n_pixels; i++) {
        if (luma[i] > max_luma)
            max_luma = luma[i];
    }
    features[0] = max_luma;
    PL_INFO(gpu, "[ML-CPU] maxscl = %.4f", max_luma);

    // Feature 1: average_maxrgb (mean luminance)
    PL_INFO(gpu, "[ML-CPU] Computing average_maxrgb (feature 1)...");
    double sum = 0.0;
    for (int i = 0; i < n_pixels; i++)
        sum += luma[i];
    features[1] = (float)(sum / n_pixels);
    PL_INFO(gpu, "[ML-CPU] average_maxrgb = %.4f", features[1]);

    // Feature 2: fraction_bright_pixels (fraction > 0.5 PQ)
    PL_INFO(gpu, "[ML-CPU] Computing fraction_bright_pixels (feature 2)...");
    int bright_count = 0;
    for (int i = 0; i < n_pixels; i++) {
        if (luma[i] > 0.5f)
            bright_count++;
    }
    features[2] = (float)bright_count / n_pixels;
    PL_INFO(gpu, "[ML-CPU] fraction_bright_pixels = %.4f", features[2]);

    // Features 3-8: Percentiles (p25, p50, p75, p90, p95, p99)
    // Use sorting + linear interpolation for accuracy (matches numpy.percentile)
    PL_INFO(gpu, "[ML-CPU] Computing percentiles (features 3-8)...");

    // Create a copy for sorting (don't modify original luma array)
    float *luma_sorted = pl_alloc(NULL, n_pixels * sizeof(float));
    if (!luma_sorted) {
        pl_free(luma);
        return false;
    }
    memcpy(luma_sorted, luma, n_pixels * sizeof(float));

    // Sort using qsort
    qsort(luma_sorted, n_pixels, sizeof(float), cmp_float);

    // Extract percentiles with linear interpolation (numpy method)
    const float targets[6] = {0.25f, 0.50f, 0.75f, 0.90f, 0.95f, 0.99f};
    for (int i = 0; i < 6; i++) {
        // Numpy-style percentile: linear interpolation between sorted values
        float pos = targets[i] * (n_pixels - 1);
        int idx_low = (int)pos;
        int idx_high = idx_low + 1;

        if (idx_high >= n_pixels) {
            features[3 + i] = luma_sorted[n_pixels - 1];
        } else {
            float frac = pos - idx_low;
            features[3 + i] = luma_sorted[idx_low] * (1.0f - frac) +
                              luma_sorted[idx_high] * frac;
        }
    }

    if (gpu_histogram) {
        uint64_t sum_bins = 0;
        int max_bin = 0, bright_bins = 0;
        for (int bin = 0; bin < 65536; bin++) {
            sum_bins += (uint64_t)bin * histogram[bin];
            if (histogram[bin]) max_bin = bin;
            if (bin > 32767) bright_bins += histogram[bin];
        }
        features[0] = max_bin / 65535.0f;
        features[1] = (float)((double)sum_bins / (65535.0 * n_pixels));
        features[2] = (float)bright_bins / n_pixels;
        const float targets[6] = {0.25f, 0.50f, 0.75f, 0.90f, 0.95f, 0.99f};
        for (int i = 0; i < 6; i++)
            features[3 + i] = histogram_percentile(histogram, n_pixels, targets[i]);
    }

    pl_free(luma_sorted);
    PL_INFO(gpu, "[ML-CPU] Percentiles complete: p50=%.4f, p90=%.4f, p99=%.4f",
            features[4], features[6], features[8]);

    float zone_stats[8][5][3] = {0};
    bool gpu_zones = extract_zone_stats_gpu(gpu, luma_tex, zone_stats);
    PL_WARN(gpu, "[ML] GPU zone statistics: %s", gpu_zones ? "enabled" : "fallback");

    // Build Summed Area Table for spatial features (fallback/reference)
    PL_INFO(gpu, "[ML-CPU] Building Summed Area Table...");
    float *sat = pl_alloc(NULL, n_pixels * sizeof(float));
    if (!sat) {
        pl_free(luma);
        return false;
    }

    // Prefix sum: rows then columns
    for (int y = 0; y < h; y++) {
        float row_sum = 0.0f;
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            row_sum += luma[idx];
            float above = (y > 0) ? sat[(y-1) * w + x] : 0.0f;
            sat[idx] = row_sum + above;
        }
    }
    PL_INFO(gpu, "[ML-CPU] SAT construction complete");

    // Helper: SAT query for rectangle sum
    #define SAT_QUERY(y0, y1, x0, x1) \
        (sat[(y1)*w + (x1)] \
         - ((y0) > 0 ? sat[((y0)-1)*w + (x1)] : 0.0f) \
         - ((x0) > 0 ? sat[(y1)*w + ((x0)-1)] : 0.0f) \
         + ((y0) > 0 && (x0) > 0 ? sat[((y0)-1)*w + ((x0)-1)] : 0.0f))

    // Features 9-17: 3×3 zone means
    // Features 18-26: 3×3 zone maxes
    PL_INFO(gpu, "[ML-CPU] Extracting 3×3 zone features (9-26)...");
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            int y0 = (r * h) / 3;
            int y1 = ((r+1) * h) / 3 - 1;
            int x0 = (c * w) / 3;
            int x1 = ((c+1) * w) / 3 - 1;

            // Mean via SAT
            float sum_rect = SAT_QUERY(y0, y1, x0, x1);
            int n_rect = (y1 - y0 + 1) * (x1 - x0 + 1);
            features[9 + r*3 + c] = gpu_zones
                ? zone_stats[c][r][0] / zone_stats[c][r][2]
                : sum_rect / n_rect;

            // Max via linear scan
            float max_rect = 0.0f;
            for (int y = y0; y <= y1; y++) {
                for (int x = x0; x <= x1; x++) {
                    float val = luma[y * w + x];
                    if (val > max_rect)
                        max_rect = val;
                }
            }
            features[18 + r*3 + c] = gpu_zones
                ? zone_stats[c][r][1]
                : max_rect;
        }
    }

    // Features 27-51: 5×5 zone means
    // Features 52-76: 5×5 zone maxes
    PL_INFO(gpu, "[ML-CPU] Extracting 5×5 zone features (27-76)...");
    for (int r = 0; r < 5; r++) {
        for (int c = 0; c < 5; c++) {
            int y0 = (r * h) / 5;
            int y1 = ((r+1) * h) / 5 - 1;
            int x0 = (c * w) / 5;
            int x1 = ((c+1) * w) / 5 - 1;

            float sum_rect = SAT_QUERY(y0, y1, x0, x1);
            int n_rect = (y1 - y0 + 1) * (x1 - x0 + 1);
            features[27 + r*5 + c] = gpu_zones
                ? zone_stats[c + 3][r][0] / zone_stats[c + 3][r][2]
                : sum_rect / n_rect;

            float max_rect = 0.0f;
            for (int y = y0; y <= y1; y++) {
                for (int x = x0; x <= x1; x++) {
                    float val = luma[y * w + x];
                    if (val > max_rect)
                        max_rect = val;
                }
            }
            features[52 + r*5 + c] = gpu_zones
                ? zone_stats[c + 3][r][1]
                : max_rect;
        }
    }

    #undef SAT_QUERY

    PL_INFO(gpu, "[ML-CPU] All zone features extracted");
    pl_free(sat);
    pl_free(luma);
    PL_INFO(gpu, "[ML-CPU] === extract_features_cpu COMPLETE ===");
    return true;
}

bool pl_extract_ml_features(pl_gpu gpu,
                             const struct pl_frame *frame,
                             const struct pl_ml_feature_params *params,
                             float features[PL_ML_FEATURE_DIM])
{
    PL_INFO(gpu, "[ML] === pl_extract_ml_features START ===");

    if (!gpu || !frame || !params || !features) {
        PL_ERR(gpu, "Invalid arguments to pl_extract_ml_features");
        return false;
    }
    PL_INFO(gpu, "[ML] Arguments validated successfully");

    memset(features, 0, PL_ML_FEATURE_DIM * sizeof(float));
    PL_INFO(gpu, "[ML] Feature buffer zeroed");

    // Feature 77: target_nits (runtime parameter)
    features[77] = params->target_nits;
    PL_INFO(gpu, "[ML] Target nits set: %.2f", params->target_nits);

    // TODO 2: Create downscaled luma texture
    PL_INFO(gpu, "[ML] Looking for r16 format...");
    pl_fmt luma_fmt = pl_find_named_fmt(gpu, "r16");
    if (!luma_fmt) {
        PL_INFO(gpu, "[ML] r16 not found, trying generic 16-bit format...");
        // Fallback: find any single-component 16-bit format
                luma_fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 16, 16,
                                                             PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_SAMPLEABLE
                                                         | PL_FMT_CAP_HOST_READABLE);
    }
    if (!luma_fmt) {
        PL_ERR(gpu, "No suitable luma format found (need 16-bit single-component)");
        return false;
    }
    PL_INFO(gpu, "[ML] Luma format found: %s", luma_fmt->name);

    PL_INFO(gpu, "[ML] Creating luma texture (%dx%d)...", params->downsample_width, params->downsample_height);
    pl_tex luma_tex = pl_tex_create(gpu, pl_tex_params(
        .w = params->downsample_width,
        .h = params->downsample_height,
        .format = luma_fmt,
        .sampleable = true,
        .renderable = true,
        .host_readable = true,
    ));

    if (!luma_tex) {
        PL_ERR(gpu, "Failed to create luma texture (%dx%d)",
               params->downsample_width, params->downsample_height);
        return false;
    }
    PL_INFO(gpu, "[ML] Luma texture created successfully");

    // Create target frame (luma-only)
    PL_INFO(gpu, "[ML] Setting up target frame (luma-only)...");
    struct pl_frame target = {
        .num_planes = 1,
        .repr = pl_color_repr_hdtv,  // BT.2020 matrix
        .color = frame->color,         // Inherit colorspace from source
    };
    target.planes[0].texture = luma_tex;
    target.planes[0].components = 1;
    target.planes[0].component_mapping[0] = PL_CHANNEL_Y;
    target.planes[0].component_mapping[1] = -1;
    target.planes[0].component_mapping[2] = -1;
    target.planes[0].component_mapping[3] = -1;
    PL_INFO(gpu, "[ML] Target frame configured");

    // Render downscaled luma
    PL_INFO(gpu, "[ML] Creating renderer...");
    pl_renderer renderer = pl_renderer_create(gpu->log, gpu);
    if (!renderer) {
        PL_ERR(gpu, "Failed to create renderer");
        pl_tex_destroy(gpu, &luma_tex);
        return false;
    }
    PL_INFO(gpu, "[ML] Renderer created successfully");

    // Check frame validity before rendering
    if (!frame || frame->num_planes == 0 || !frame->planes[0].texture) {
        PL_ERR(gpu, "[ML] Invalid frame: num_planes=%d", frame ? frame->num_planes : 0);
        pl_renderer_destroy(&renderer);
        pl_tex_destroy(gpu, &luma_tex);
        return false;
    }

    PL_INFO(gpu, "[ML] Rendering downscaled luma (input: %dx%d -> output: %dx%d)...",
            frame->planes[0].texture->params.w, frame->planes[0].texture->params.h,
            params->downsample_width, params->downsample_height);
    bool ok = pl_render_image(renderer, frame, &target, &pl_render_default_params);
    pl_renderer_destroy(&renderer);

    if (!ok) {
        PL_ERR(gpu, "pl_render_image failed");
        pl_tex_destroy(gpu, &luma_tex);
        return false;
    }
    PL_INFO(gpu, "[ML] Luma rendering complete");

    if (!params->force_cpu_fallback && !params->debug_luma_path &&
        extract_features_gpu(gpu, luma_tex, features)) {
        PL_INFO(gpu, "[ML] GPU feature extraction complete (no luma readback)");
        pl_tex_destroy(gpu, &luma_tex);
        return true;
    }

    // Extract features from luma texture (CPU fallback)
    PL_INFO(gpu, "[ML] Starting CPU feature extraction...");
    ok = extract_features_cpu(gpu, luma_tex, params, features);

    pl_tex_destroy(gpu, &luma_tex);

    if (ok) {
        PL_INFO(gpu, "[ML] === Feature extraction COMPLETE (success) ===");
        PL_INFO(gpu, "[ML] Sample features: maxscl=%.4f, avg=%.4f, bright_frac=%.4f",
                features[0], features[1], features[2]);
    } else {
        PL_ERR(gpu, "[ML] === Feature extraction FAILED ===");
    }

    return ok;
}

void pl_reset_ml_feature_state(pl_gpu gpu)
{
    // TODO: Clear any cached textures/buffers
    // For now: no-op (stateless implementation)
}
