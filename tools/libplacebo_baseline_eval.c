/*
 * libplacebo_baseline_eval.c
 * ==========================
 * Evaluates libplacebo's pl_tone_map_spline for each scene and outputs
 * the 256-point curve in PQ signal space [0, 1].
 *
 * Design decisions (fixed):
 *   - Target display: 143 nits (no-tier training — polynomial is tier-independent)
 *   - Evaluation domain: x = linspace(0, 1, 256) — full PQ range, same as gold curves
 *   - Scene peak: l1_max_pq from RPU L1 metadata (NOT GPU-derived maxscl)
 *     Reason: DV P5 decoded without native DV support produces wrong chroma
 *     (green/purple cast). RPU metadata is read directly from the bitstream
 *     and is always correct. For luma tone mapping the I channel is unaffected,
 *     but we use RPU metadata to be safe and consistent.
 *   - For x > l1_max_pq: pl_tone_map_sample returns target_pq (clips to output max)
 *     This is valid — the gold curve also continues the polynomial above the scene
 *     peak, and both are compared in the same [0,1] PQ domain in BoundedDTMLoss.
 *
 * Input  (stdin):  CSV with header: scene_id,l1_max_pq,l1_avg_pq
 * Output (stdout): CSV:             scene_id,spline_y_0,...,spline_y_255
 *
 * Build: see tools/meson.build (libplacebo_baseline_eval target)
 * Run:   python3 tools/gen_spline_baselines.py
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <libplacebo/tone_mapping.h>
#include <libplacebo/colorspace.h>

#define N_PTS         256
#define TARGET_NITS   143.0f   /* fixed: no-tier, 143-nit reference display */

/* ST.2084 (PQ) forward transform: linear light (nits/10000) -> PQ signal */
static float nits_to_pq(float nits)
{
    float L  = nits / 10000.0f;
    float m1 = 0.1593017578125f, m2 = 78.84375f;
    float c1 = 0.8359375f, c2 = 18.8515625f, c3 = 18.6875f;
    if (L <= 0.0f) return 0.0f;
    float Lm1 = powf(L, m1);
    return powf((c1 + c2 * Lm1) / (1.0f + c3 * Lm1), m2);
}

int main(void)
{
    char line[4096];
    const float target_pq = nits_to_pq(TARGET_NITS);

    /* Skip CSV header */
    if (!fgets(line, sizeof(line), stdin))
        return 1;

    /* Print output header */
    printf("scene_id");
    for (int i = 0; i < N_PTS; i++)
        printf(",spline_y_%d", i);
    printf("\n");

    int scene_id;
    float l1_max_pq, l1_avg_pq;

    while (fgets(line, sizeof(line), stdin)) {
        /* Parse: scene_id,l1_max_pq,l1_avg_pq */
        if (sscanf(line, "%d,%f,%f", &scene_id, &l1_max_pq, &l1_avg_pq) != 3)
            continue;

        /* Clamp to valid PQ range — use RPU L1 values directly */
        if (l1_max_pq < 0.001f) l1_max_pq = 0.001f;
        if (l1_max_pq > 1.0f  ) l1_max_pq = 1.0f;
        if (l1_avg_pq < 0.0f  ) l1_avg_pq = 0.0f;
        if (l1_avg_pq > l1_max_pq) l1_avg_pq = l1_max_pq;

        struct pl_tone_map_params params = {
            .function       = &pl_tone_map_spline,
            .constants      = { PL_TONE_MAP_CONSTANTS },
            .input_scaling  = PL_HDR_PQ,
            .output_scaling = PL_HDR_PQ,
            .input_min      = 0.0f,
            .input_max      = l1_max_pq,   /* scene peak from RPU metadata */
            .input_avg      = l1_avg_pq,   /* scene avg from RPU metadata */
            .output_min     = 0.0f,
            .output_max     = target_pq,   /* 143 nits, fixed */
            .lut_size       = N_PTS,
        };

        pl_tone_map_params_infer(&params);

        /*
         * Evaluate at N_PTS evenly spaced points over [0, 1] — full PQ range.
         * Same domain as gold RPU polynomial curves in training.
         * For x > l1_max_pq: pl_tone_map_sample clips to target_pq (valid).
         */
        printf("%d", scene_id);
        for (int i = 0; i < N_PTS; i++) {
            float x = (float)i / (float)(N_PTS - 1);
            float y = pl_tone_map_sample(x, &params);
            if (y < 0.0f) y = 0.0f;
            if (y > 1.0f) y = 1.0f;
            printf(",%f", y);
        }
        printf("\n");
    }

    return 0;
}
