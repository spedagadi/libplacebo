/*
 * libplacebo_baseline_eval.c
 * ==========================
 * Evaluates libplacebo's pl_tone_map_spline curve for each scene in the
 * val dataset and outputs the 256-point curve in PQ signal space.
 *
 * Used to compare the ML model's predicted curves against libplacebo's
 * built-in adaptive spline tone mapper.
 *
 * Input  (stdin):  CSV with header: scene_id,maxscl,l1_avg_pq,target_nits
 * Output (stdout): CSV: scene_id,y_0,y_1,...,y_255
 *
 * Both input and output curves are in PQ signal space [0,1].
 * The 256 points are evaluated at x = linspace(0, maxscl, 256) so both
 * libplacebo and the gold RPU polynomial are compared on the same domain
 * (the meaningful content range up to the scene peak).
 *
 * Build: see tools/meson.build (libplacebo_baseline_eval target)
 * Run:   python3 tools/libplacebo_baseline_eval.py
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <libplacebo/tone_mapping.h>
#include <libplacebo/colorspace.h>

#define N_PTS 256

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

    /* Skip CSV header */
    if (!fgets(line, sizeof(line), stdin))
        return 1;

    /* Print output header */
    printf("scene_id");
    for (int i = 0; i < N_PTS; i++)
        printf(",y_%d", i);
    printf("\n");

    int scene_id;
    float maxscl, l1_avg_pq, target_nits;

    while (fgets(line, sizeof(line), stdin)) {
        /* Parse: scene_id,maxscl,l1_avg_pq,target_nits */
        if (sscanf(line, "%d,%f,%f,%f",
                   &scene_id, &maxscl, &l1_avg_pq, &target_nits) != 4)
            continue;

        /* Clamp inputs to valid PQ range */
        if (maxscl    < 0.001f) maxscl    = 0.001f;
        if (maxscl    > 1.0f  ) maxscl    = 1.0f;
        if (l1_avg_pq < 0.0f  ) l1_avg_pq = 0.0f;
        if (l1_avg_pq > maxscl) l1_avg_pq = maxscl;

        float target_pq = nits_to_pq(target_nits);

        struct pl_tone_map_params params = {
            .function       = &pl_tone_map_spline,
            .constants      = { PL_TONE_MAP_CONSTANTS },
            .input_scaling  = PL_HDR_PQ,
            .output_scaling = PL_HDR_PQ,
            .input_min      = 0.0f,
            .input_max      = maxscl,
            .input_avg      = l1_avg_pq,
            .output_min     = 0.0f,
            .output_max     = target_pq,
            .lut_size       = N_PTS,
        };

        pl_tone_map_params_infer(&params);

        /* Sample at N_PTS evenly spaced points over [0, maxscl] */
        printf("%d", scene_id);
        for (int i = 0; i < N_PTS; i++) {
            float x = maxscl * (float)i / (float)(N_PTS - 1);
            float y = pl_tone_map_sample(x, &params);
            /* Clamp to [0,1] — output is in PQ units */
            if (y < 0.0f) y = 0.0f;
            if (y > 1.0f) y = 1.0f;
            printf(",%f", y);
        }
        printf("\n");
    }

    return 0;
}
