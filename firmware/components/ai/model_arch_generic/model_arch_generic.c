#include "model_arch_generic.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"

static const char* TAG = "MODEL_GENERIC";

static bool generic_interface_run(anomaly_model_t* model, const float* input_data, size_t input_len, const fft_shared_data_t* shared_fft, float* out_anomaly_score) {
    if (!model || !model->instance_data || !input_data || !out_anomaly_score) return false;
    model_arch_generic_impl_t* impl = (model_arch_generic_impl_t*)model->instance_data;

    tflite_tensor_details_t* in_det = &impl->base.input_details;
    tflite_tensor_details_t* out_det = &impl->base.output_details;

    // Copy input to model input tensor
    size_t copy_bytes = (input_len * sizeof(float) < in_det->bytes) ? input_len * sizeof(float) : in_det->bytes;
    memcpy(in_det->data, input_data, copy_bytes);

    if (!base_model_run(&impl->base)) {
        ESP_LOGE(TAG, "Failed to run generic model");
        return false;
    }

    if (impl->scoring_mode == GENERIC_SCORE_DIRECT) {
        // Direct value: return the first output float (e.g. classifier probability)
        float* outputs = (float*)out_det->data;
        if (out_det->bytes >= sizeof(float) && outputs != NULL) {
            *out_anomaly_score = outputs[0];
        } else {
            *out_anomaly_score = 0.0f;
            ESP_LOGW(TAG, "Output tensor does not contain float data");
        }
    } else {
        // Reconstruction MSE: compute MSE between input and output tensors
        float* rec_out = (float*)out_det->data;
        float* original = (float*)in_det->data;
        size_t elements = copy_bytes / sizeof(float);

        if (elements == 0 || rec_out == NULL) {
            *out_anomaly_score = 0.0f;
            return false;
        }

        float sum_sq = 0.0f;
        for (size_t i = 0; i < elements; i++) {
            float diff = original[i] - rec_out[i];
            sum_sq += diff * diff;
        }
        float mse = sum_sq / (float)elements;
        *out_anomaly_score = mse;
    }

    // Apply linear scaling multiplier
    *out_anomaly_score *= impl->score_multiplier;
    return true;
}

static void generic_interface_deinit(anomaly_model_t* model) {
    if (!model || !model->instance_data) return;
    model_arch_generic_impl_t* impl = (model_arch_generic_impl_t*)model->instance_data;
    base_model_deinit(&impl->base);
    free(impl);
    model->instance_data = NULL;
}

static bool generic_interface_update(anomaly_model_t* model, const uint8_t* new_model_bytes) {
    if (!model || !model->instance_data || !new_model_bytes) return false;
    model_arch_generic_impl_t* impl = (model_arch_generic_impl_t*)model->instance_data;
    return base_model_reload(&impl->base, new_model_bytes);
}

bool model_arch_generic_anomaly_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, generic_scoring_mode_t scoring_mode, float multiplier) {
    if (!model || !model_bytes || !config) return false;
    model_arch_generic_impl_t* impl = (model_arch_generic_impl_t*)calloc(1, sizeof(model_arch_generic_impl_t));
    if (!impl) return false;

    impl->scoring_mode = scoring_mode;
    impl->score_multiplier = multiplier <= 0.0f ? 1.0f : multiplier;

    if (!base_model_init(&impl->base, model_bytes, config)) {
        free(impl);
        return false;
    }

    model->name = "Generic TFLite Model Wrapper";
    model->model_id = (uint32_t)(uintptr_t)impl;
    model->type_id = 99; // Arbitrary type ID for generic custom models
    model->instance_data = impl;
    model->run = generic_interface_run;
    model->deinit = generic_interface_deinit;
    model->update = generic_interface_update;

    return true;
}
