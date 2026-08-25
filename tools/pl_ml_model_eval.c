#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <libplacebo/log.h>
#include <libplacebo/ml_model.h>

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s MODEL FEATURES_BIN\n", argv[0]);
        return 2;
    }

    pl_log log = pl_log_create(PL_API_VER, pl_log_params(
        .log_cb = pl_log_simple, .log_priv = stderr, .log_level = PL_LOG_WARN));
    pl_ml_model model = pl_ml_model_create(log, pl_ml_model_params(.path = argv[1]));
    if (!model) {
        pl_log_destroy(&log);
        return 1;
    }

    float features[88];
    FILE *file = fopen(argv[2], "rb");
    if (!file || fread(features, sizeof(features), 1, file) != 1) {
        fprintf(stderr, "Failed reading 88 features from %s\n", argv[2]);
        if (file) fclose(file);
        pl_ml_model_destroy(&model);
        pl_log_destroy(&log);
        return 1;
    }
    fclose(file);

    struct pl_ml_prediction prediction;
    bool ok = pl_ml_model_predict(model, features, 88, &prediction);
    if (ok) printf("%.9f\n", prediction.gamma);
    pl_ml_model_destroy(&model);
    pl_log_destroy(&log);
    return ok ? 0 : 1;
}
