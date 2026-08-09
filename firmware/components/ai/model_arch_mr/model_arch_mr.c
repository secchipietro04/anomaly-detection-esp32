#include "model_arch_mr.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"

static const char* TAG = "MODEL_MR";

// Helper function to execute MiniRocket core inference
static bool run_mr_core(model_arch_mr_impl_t* impl, const float* input_series, size_t input_len, float* out_probabilities) {
    if (impl->use_tflite) {
        tflite_tensor_details_t* in_det = &impl->base.input_details;
        tflite_tensor_details_t* out_det = &impl->base.output_details;

        size_t copy_bytes = (input_len * sizeof(float) < in_det->bytes) ? input_len * sizeof(float) : in_det->bytes;
        memcpy(in_det->data, input_series, copy_bytes);

        if (!base_model_run(&impl->base)) {
            return false;
        }

        if (out_det->bytes > 0 && out_det->data != NULL) {
            memcpy(out_probabilities, out_det->data, out_det->bytes);
            return true;
        }
        return false;
    } else {
        size_t L = impl->config.input_length;
        size_t num_feats = impl->config.num_features;
        size_t num_classes = impl->config.num_classes;

        if (num_feats == 0 || num_classes == 0 || !impl->feature_buffer) return false;

        // Perform MiniRocket Transform
        for (size_t f = 0; f < num_feats; f++) {
            size_t kernel_idx = f % impl->config.num_kernels;
            int32_t d = impl->config.dilations[f];
            float bias = impl->config.biases[f];

            const int8_t* k_weights = &impl->config.kernels[kernel_idx * 9];

            size_t count = 0;
            size_t total = 0;

            int padding = (8 * d) / 2;
            for (int i = 0; i < (int)L; i++) {
                float sum = 0;
                for (int j = 0; j < 9; j++) {
                    int idx = i - padding + j * d;
                    if (idx >= 0 && idx < (int)L) {
                        sum += input_series[idx] * k_weights[j];
                    }
                }
                if (sum > bias) {
                    count++;
                }
                total++;
            }

            impl->feature_buffer[f] = (float)count / total;
        }

        // Apply Linear Decision Layer
        for (size_t c = 0; c < num_classes; c++) {
            float val = impl->config.intercepts[c];
            for (size_t f = 0; f < num_feats; f++) {
                val += impl->feature_buffer[f] * impl->config.weights[c * num_feats + f];
            }
            out_probabilities[c] = val;
        }

        // Apply Softmax
        float max_val = out_probabilities[0];
        for (size_t c = 1; c < num_classes; c++) {
            if (out_probabilities[c] > max_val) {
                max_val = out_probabilities[c];
            }
        }

        float sum_exp = 0.0f;
        for (size_t c = 0; c < num_classes; c++) {
            out_probabilities[c] = expf(out_probabilities[c] - max_val);
            sum_exp += out_probabilities[c];
        }

        for (size_t c = 0; c < num_classes; c++) {
            out_probabilities[c] /= sum_exp;
        }

        return true;
    }
}

// ==========================================
// Anomaly Detection Interface Implementation
// ==========================================
static bool mr_anomaly_run(anomaly_model_t* model, const float* input_data, size_t input_len, const fft_shared_data_t* shared_fft, float* out_anomaly_score) {
    if (!model || !model->instance_data || !input_data || !out_anomaly_score) return false;
    model_arch_mr_impl_t* impl = (model_arch_mr_impl_t*)model->instance_data;

    size_t num_classes = impl->use_tflite ? (impl->base.output_details.bytes / sizeof(float)) : impl->config.num_classes;
    if (num_classes == 0) return false;

    float* probs = (float*)malloc(num_classes * sizeof(float));
    if (!probs) return false;

    if (!run_mr_core(impl, input_data, input_len, probs)) {
        free(probs);
        return false;
    }

    size_t target_class = impl->config.anomaly_class_idx;
    if (target_class >= num_classes) {
        target_class = num_classes - 1; // Safeguard bounds
    }

    *out_anomaly_score = probs[target_class];
    free(probs);
    return true;
}

static void mr_anomaly_deinit(anomaly_model_t* model) {
    if (!model || !model->instance_data) return;
    model_arch_mr_impl_t* impl = (model_arch_mr_impl_t*)model->instance_data;
    if (impl->use_tflite) {
        base_model_deinit(&impl->base);
    } else {
        free(impl->feature_buffer);
    }
    free(impl);
    model->instance_data = NULL;
}

static bool mr_anomaly_update(anomaly_model_t* model, const uint8_t* new_model_bytes) {
    if (!model || !model->instance_data || !new_model_bytes) return false;
    model_arch_mr_impl_t* impl = (model_arch_mr_impl_t*)model->instance_data;
    if (impl->use_tflite) {
        return base_model_reload(&impl->base, new_model_bytes);
    }
    return false; // Custom C models cannot be updated via flatbuffer reloading
}

bool model_arch_mr_anomaly_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, const model_arch_mr_config_t* mr_cfg) {
    if (!model) return false;
    model_arch_mr_impl_t* impl = (model_arch_mr_impl_t*)calloc(1, sizeof(model_arch_mr_impl_t));
    if (!impl) return false;

    if (model_bytes != NULL && config != NULL) {
        impl->use_tflite = true;
        if (!base_model_init(&impl->base, model_bytes, config)) {
            free(impl);
            return false;
        }
    } else if (mr_cfg != NULL) {
        impl->use_tflite = false;
        impl->config = *mr_cfg;
        if (impl->config.num_features > 0) {
            impl->feature_buffer = (float*)malloc(impl->config.num_features * sizeof(float));
            if (!impl->feature_buffer) {
                free(impl);
                return false;
            }
        }
    } else {
        free(impl);
        return false;
    }

    model->name = "MiniRocket Anomaly Detection Model";
    model->model_id = (uint32_t)(uintptr_t)impl;
    model->type_id = ModelType_model_router_MR_m_c; // Shared ID mapping
    model->instance_data = impl;
    model->run = mr_anomaly_run;
    model->deinit = mr_anomaly_deinit;
    model->update = mr_anomaly_update;

    return true;
}

// ==========================================
// Router Interface Implementation
// ==========================================
static bool mr_router_run(router_model_t* router, const float* input_data, size_t input_len, int* out_best_route_idx, float* out_confidence, fft_shared_data_t* out_shared_fft) {
    if (!router || !router->instance_data || !input_data || !out_best_route_idx || !out_confidence) return false;
    model_arch_mr_impl_t* impl = (model_arch_mr_impl_t*)router->instance_data;

    size_t num_classes = impl->use_tflite ? (impl->base.output_details.bytes / sizeof(float)) : impl->config.num_classes;
    if (num_classes == 0) return false;

    float* probs = (float*)malloc(num_classes * sizeof(float));
    if (!probs) return false;

    if (!run_mr_core(impl, input_data, input_len, probs)) {
        free(probs);
        return false;
    }

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
    free(probs);

    // MiniRocket does not produce FFT results
    if (out_shared_fft) {
        out_shared_fft->is_valid = false;
    }

    return true;
}

static void mr_router_deinit(router_model_t* router) {
    if (!router || !router->instance_data) return;
    model_arch_mr_impl_t* impl = (model_arch_mr_impl_t*)router->instance_data;
    if (impl->use_tflite) {
        base_model_deinit(&impl->base);
    } else {
        free(impl->feature_buffer);
    }
    free(impl);
    router->instance_data = NULL;
}

static bool mr_router_update(router_model_t* router, const uint8_t* new_model_bytes) {
    if (!router || !router->instance_data || !new_model_bytes) return false;
    model_arch_mr_impl_t* impl = (model_arch_mr_impl_t*)router->instance_data;
    if (impl->use_tflite) {
        return base_model_reload(&impl->base, new_model_bytes);
    }
    return false;
}

bool model_arch_mr_router_init(router_model_t* router, const uint8_t* model_bytes, const base_model_config_t* config, const model_arch_mr_config_t* mr_cfg) {
    if (!router) return false;
    model_arch_mr_impl_t* impl = (model_arch_mr_impl_t*)calloc(1, sizeof(model_arch_mr_impl_t));
    if (!impl) return false;

    if (model_bytes != NULL && config != NULL) {
        impl->use_tflite = true;
        if (!base_model_init(&impl->base, model_bytes, config)) {
            free(impl);
            return false;
        }
    } else if (mr_cfg != NULL) {
        impl->use_tflite = false;
        impl->config = *mr_cfg;
        if (impl->config.num_features > 0) {
            impl->feature_buffer = (float*)malloc(impl->config.num_features * sizeof(float));
            if (!impl->feature_buffer) {
                free(impl);
                return false;
            }
        }
    } else {
        free(impl);
        return false;
    }

    router->name = "MiniRocket Router";
    router->model_id = (uint32_t)(uintptr_t)impl;
    router->type_id = ModelType_model_router_MR_m_c;
    router->instance_data = impl;
    router->input_length = (mr_cfg != NULL) ? mr_cfg->input_length : 0;
    router->run = mr_router_run;
    router->deinit = mr_router_deinit;
    router->update = mr_router_update;

    return true;
}
