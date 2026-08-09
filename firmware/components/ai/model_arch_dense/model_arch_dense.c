#include "model_arch_dense.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"

static const char* TAG = "MODEL_DENSE";

// ==========================================
// Anomaly Detection Interface Implementation
// ==========================================
static bool dense_anomaly_run(anomaly_model_t* model, const float* input_data, size_t input_len, const fft_shared_data_t* shared_fft, float* out_anomaly_score) {
    if (!model || !model->instance_data || !input_data || !out_anomaly_score) return false;
    model_arch_dense_impl_t* impl = (model_arch_dense_impl_t*)model->instance_data;

    tflite_tensor_details_t* in_det = &impl->base.input_details;
    tflite_tensor_details_t* out_det = &impl->base.output_details;

    size_t copy_bytes = (input_len * sizeof(float) < in_det->bytes) ? input_len * sizeof(float) : in_det->bytes;
    memcpy(in_det->data, input_data, copy_bytes);

    if (!base_model_run(&impl->base)) {
        ESP_LOGE(TAG, "Failed to run Dense Anomaly Model");
        return false;
    }

    // Compute raw MSE reconstruction score
    float* rec_out = (float*)out_det->data;
    float* original = (float*)in_det->data;
    size_t elements = copy_bytes / sizeof(float);

    float sum_sq = 0.0f;
    for (size_t i = 0; i < elements; i++) {
        float diff = original[i] - rec_out[i];
        sum_sq += diff * diff;
    }
    float mse = sum_sq / (float)elements;

    *out_anomaly_score = impl->score_gamma * mse;
    return true;
}

static void dense_anomaly_deinit(anomaly_model_t* model) {
    if (!model || !model->instance_data) return;
    model_arch_dense_impl_t* impl = (model_arch_dense_impl_t*)model->instance_data;
    base_model_deinit(&impl->base);
    free(impl);
    model->instance_data = NULL;
}

static bool dense_anomaly_update(anomaly_model_t* model, const uint8_t* new_model_bytes) {
    if (!model || !model->instance_data || !new_model_bytes) return false;
    model_arch_dense_impl_t* impl = (model_arch_dense_impl_t*)model->instance_data;
    return base_model_reload(&impl->base, new_model_bytes);
}

bool model_arch_dense_anomaly_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, float gamma) {
    if (!model || !model_bytes || !config) return false;
    model_arch_dense_impl_t* impl = (model_arch_dense_impl_t*)calloc(1, sizeof(model_arch_dense_impl_t));
    if (!impl) return false;

    impl->score_gamma = gamma <= 0.0f ? 1.0f : gamma;

    if (!base_model_init(&impl->base, model_bytes, config)) {
        free(impl);
        return false;
    }

    model->name = "Dense Autoencoder (DA)";
    model->model_id = (uint32_t)(uintptr_t)impl;
    model->type_id = ModelType_model_arch_DA_m_c;
    model->instance_data = impl;
    model->run = dense_anomaly_run;
    model->deinit = dense_anomaly_deinit;
    model->update = dense_anomaly_update;

    return true;
}

// ==========================================
// Router Interface Implementation
// ==========================================
static bool dense_router_run(router_model_t* router, const float* input_data, size_t input_len, int* out_best_route_idx, float* out_confidence, fft_shared_data_t* out_shared_fft) {
    if (!router || !router->instance_data || !input_data || !out_best_route_idx || !out_confidence) return false;
    model_arch_dense_impl_t* impl = (model_arch_dense_impl_t*)router->instance_data;

    tflite_tensor_details_t* in_det = &impl->base.input_details;
    tflite_tensor_details_t* out_det = &impl->base.output_details;

    size_t copy_bytes = (input_len * sizeof(float) < in_det->bytes) ? input_len * sizeof(float) : in_det->bytes;
    memcpy(in_det->data, input_data, copy_bytes);

    if (!base_model_run(&impl->base)) {
        ESP_LOGE(TAG, "Failed to run Dense Router Model");
        return false;
    }

    // Find class with the maximum probability (ArgMax)
    float* probs = (float*)out_det->data;
    size_t num_classes = out_det->bytes / sizeof(float);
    if (num_classes == 0) return false;

    int max_idx = 0;
    float max_prob = probs[0];
    for (size_t i = 1; i < num_classes; i++) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            max_idx = i;
        }
    }

    *out_best_route_idx = max_idx;
    *out_confidence = max_prob;

    if (out_shared_fft) {
        out_shared_fft->is_valid = false;
    }

    return true;
}

static void dense_router_deinit(router_model_t* router) {
    if (!router || !router->instance_data) return;
    model_arch_dense_impl_t* impl = (model_arch_dense_impl_t*)router->instance_data;
    base_model_deinit(&impl->base);
    free(impl);
    router->instance_data = NULL;
}

static bool dense_router_update(router_model_t* router, const uint8_t* new_model_bytes) {
    if (!router || !router->instance_data || !new_model_bytes) return false;
    model_arch_dense_impl_t* impl = (model_arch_dense_impl_t*)router->instance_data;
    return base_model_reload(&impl->base, new_model_bytes);
}

bool model_arch_dense_router_init(router_model_t* router, const uint8_t* model_bytes, const base_model_config_t* config, size_t input_length) {
    if (!router || !model_bytes || !config) return false;
    model_arch_dense_impl_t* impl = (model_arch_dense_impl_t*)calloc(1, sizeof(model_arch_dense_impl_t));
    if (!impl) return false;

    if (!base_model_init(&impl->base, model_bytes, config)) {
        free(impl);
        return false;
    }

    router->name = "Dense Router";
    router->model_id = (uint32_t)(uintptr_t)impl;
    router->type_id = ModelType_model_router_dense_m_c;
    router->instance_data = impl;
    router->input_length = input_length;
    router->run = dense_router_run;
    router->deinit = dense_router_deinit;
    router->update = dense_router_update;

    return true;
}
