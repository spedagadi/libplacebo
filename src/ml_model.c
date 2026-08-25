#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "common.h"
#include "log.h"
#include <libplacebo/ml_model.h>

#define PL_ML_MODEL_MAGIC 0x47584c50u
#define PL_ML_MODEL_VERSION 1u

typedef struct {
    int32_t left;
    int32_t right;
    int32_t feature;
    float threshold;
    uint8_t default_left;
    uint8_t split_type;
} ModelNode;

typedef struct {
    uint32_t node_count;
    ModelNode *nodes;
    float *weights;
} ModelTree;

struct pl_ml_model_t {
    uint32_t feature_count;
    uint32_t tree_count;
    float base_score;
    float learning_rate;
    ModelTree *trees;
};

struct pl_ml_context_t {
    pl_ml_model model;
};

static bool read_exact(FILE *file, void *data, size_t size)
{
    return fread(data, 1, size, file) == size;
}

pl_ml_model pl_ml_model_create(pl_log log, const struct pl_ml_model_params *params)
{
    if (!params || !params->path) return NULL;
    FILE *file = fopen(params->path, "rb");
    if (!file) {
        pl_msg(log, PL_LOG_ERR, "Failed opening ML model: %s", params->path);
        return NULL;
    }

    uint32_t magic, version, reserved;
    pl_ml_model model = calloc(1, sizeof(*model));
    if (!model || !read_exact(file, &magic, 4) || !read_exact(file, &version, 4) ||
        !read_exact(file, &model->feature_count, 4) ||
        !read_exact(file, &model->tree_count, 4) || !read_exact(file, &reserved, 4) ||
        !read_exact(file, &model->base_score, 4) ||
        !read_exact(file, &model->learning_rate, 4) ||
        magic != PL_ML_MODEL_MAGIC || version != PL_ML_MODEL_VERSION ||
        model->feature_count == 0 || model->tree_count == 0) {
        pl_msg(log, PL_LOG_ERR, "Invalid libplacebo ML model: %s", params->path);
        free(model); fclose(file); return NULL;
    }

    model->trees = calloc(model->tree_count, sizeof(*model->trees));
    if (!model->trees) { free(model); fclose(file); return NULL; }
    for (uint32_t t = 0; t < model->tree_count; t++) {
        ModelTree *tree = &model->trees[t];
        if (!read_exact(file, &tree->node_count, 4) || tree->node_count == 0) goto fail;
        tree->nodes = calloc(tree->node_count, sizeof(*tree->nodes));
        tree->weights = calloc(tree->node_count, sizeof(*tree->weights));
        if (!tree->nodes || !tree->weights) goto fail;
        for (uint32_t n = 0; n < tree->node_count; n++) {
            if (!read_exact(file, &tree->nodes[n].left, 4) ||
                !read_exact(file, &tree->nodes[n].right, 4) ||
                !read_exact(file, &tree->nodes[n].feature, 4) ||
                !read_exact(file, &tree->nodes[n].threshold, 4) ||
                !read_exact(file, &tree->nodes[n].default_left, 1) ||
                !read_exact(file, &tree->nodes[n].split_type, 1)) goto fail;
        }
        if (!read_exact(file, tree->weights, tree->node_count * sizeof(float))) goto fail;
    }
    fclose(file);
    return model;

fail:
    pl_ml_model_destroy(&model);
    fclose(file);
    return NULL;
}

bool pl_ml_model_predict(pl_ml_model model, const float *features, int feature_count,
                         struct pl_ml_prediction *prediction)
{
    if (!model || !features || !prediction || feature_count != (int)model->feature_count)
        return false;
    double result = model->base_score;
    for (uint32_t t = 0; t < model->tree_count; t++) {
        const ModelTree *tree = &model->trees[t];
        int32_t node = 0;
        for (uint32_t steps = 0; steps < tree->node_count; steps++) {
            const ModelNode *current = &tree->nodes[node];
            if (current->left < 0 || current->right < 0) {
                /* XGBoost's exported base_weights already contain the
                 * effective leaf contribution; do not apply learning_rate. */
                result += tree->weights[node];
                break;
            }
            if (current->feature < 0 || current->feature >= feature_count) return false;
            node = features[current->feature] < current->threshold
                ? current->left : current->right;
            if (node < 0 || node >= (int32_t)tree->node_count) return false;
        }
    }
    prediction->gamma = fmaxf(0.5f, fminf(1.5f, (float)result));
    return true;
}

void pl_ml_model_destroy(pl_ml_model *model)
{
    if (!model || !*model) return;
    for (uint32_t t = 0; t < (*model)->tree_count; t++) {
        free((*model)->trees[t].nodes);
        free((*model)->trees[t].weights);
    }
    free((*model)->trees);
    free(*model);
    *model = NULL;
}

pl_ml_context pl_ml_context_create(const struct pl_ml_context_params *params)
{
    if (!params || !params->model_path)
        return NULL;

    pl_ml_context context = calloc(1, sizeof(*context));
    if (!context)
        return NULL;

    context->model = pl_ml_model_create(params->log,
                                        pl_ml_model_params(.path = params->model_path));
    if (!context->model) {
        free(context);
        return NULL;
    }
    return context;
}

bool pl_ml_context_predict(pl_ml_context context, const float *features,
                           int feature_count,
                           struct pl_ml_prediction *prediction)
{
    return context && pl_ml_model_predict(context->model, features,
                                           feature_count, prediction);
}

void pl_ml_context_destroy(pl_ml_context *context)
{
    if (!context || !*context)
        return;
    pl_ml_model_destroy(&(*context)->model);
    free(*context);
    *context = NULL;
}
