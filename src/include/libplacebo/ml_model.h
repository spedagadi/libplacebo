#ifndef LIBPLACEBO_ML_MODEL_H_
#define LIBPLACEBO_ML_MODEL_H_

#include <libplacebo/common.h>
#include <libplacebo/log.h>

PL_API_BEGIN

typedef struct pl_ml_model_t *pl_ml_model;
typedef struct pl_ml_context_t *pl_ml_context;

struct pl_ml_model_params {
    const char *path;
};

#define pl_ml_model_params(...) (&(struct pl_ml_model_params) { __VA_ARGS__ })

struct pl_ml_prediction {
    float gamma;
};

struct pl_ml_context_params {
    pl_log log;
    const char *model_path;
};

#define pl_ml_context_params(...) (&(struct pl_ml_context_params) { __VA_ARGS__ })

PL_API pl_ml_model pl_ml_model_create(pl_log log,
                                      const struct pl_ml_model_params *params);
PL_API bool pl_ml_model_predict(pl_ml_model model,
                                const float *features, int feature_count,
                                struct pl_ml_prediction *prediction);
PL_API void pl_ml_model_destroy(pl_ml_model *model);

/* Reusable inference context. The model is loaded once at creation and can
 * be used for predictions across frames. */
PL_API pl_ml_context pl_ml_context_create(
    const struct pl_ml_context_params *params);
PL_API bool pl_ml_context_predict(pl_ml_context context,
                                  const float *features, int feature_count,
                                  struct pl_ml_prediction *prediction);
PL_API void pl_ml_context_destroy(pl_ml_context *context);

PL_API_END

#endif
