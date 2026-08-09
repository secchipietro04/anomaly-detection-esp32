#include "model_arch_ca.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "base_model.h"

static const char* TAG = "MODEL_CA";

static bool ca_interface_run(anomaly_model_t* model, const float* input_data, size_t input_len, const fft_shared_data_t* shared_fft, float* out_anomaly_score) {
    if (!model || !model->instance_data || !input_data || !out_anomaly_score) return false;
    model_arch_ca_impl_t* impl = (model_arch_ca_impl_t*)model->instance_data;

    tflite_tensor_details_t* in_det = &impl->base.input_details;
    tflite_tensor_details_t* out_det = &impl->base.output_details;

    // Copy input to model input tensor
    size_t copy_bytes = (input_len * sizeof(float) < in_det->bytes) ? input_len * sizeof(float) : in_det->bytes;
    memcpy(in_det->data, input_data, copy_bytes);

    if (!base_model_run(&impl->base)) {
        ESP_LOGE(TAG, "Failed to run Convolutional Autoencoder");
        return false;
    }

    // Compute raw reconstruction MSE
    float* rec_out = (float*)out_det->data;
    float* original = (float*)in_det->data;
    size_t elements = copy_bytes / sizeof(float);

    float sum_sq = 0.0f;
    for (size_t i = 0; i < elements; i++) {
        float diff = original[i] - rec_out[i];
        sum_sq += diff * diff;
    }
    float mse = sum_sq / (float)elements;

    
    *out_anomaly_score = mse;
    return true;
}

static void ca_interface_deinit(anomaly_model_t* model) {
    if (!model || !model->instance_data) return;
    model_arch_ca_impl_t* impl = (model_arch_ca_impl_t*)model->instance_data;
    base_model_deinit(&impl->base);
    free(impl);
    model->instance_data = NULL;
}

static bool ca_interface_update(anomaly_model_t* model, const uint8_t* new_model_bytes) {
    if (!model || !model->instance_data || !new_model_bytes) return false;
    model_arch_ca_impl_t* impl = (model_arch_ca_impl_t*)model->instance_data;
    return base_model_reload(&impl->base, new_model_bytes);
}

bool model_arch_ca_anomaly_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, float gamma) {
    if (!model || !model_bytes || !config) return false;
    model_arch_ca_impl_t* impl = (model_arch_ca_impl_t*)calloc(1, sizeof(model_arch_ca_impl_t));
    if (!impl) return false;

    impl->score_gamma = gamma <= 0.0f ? 1.0f : gamma;

    if (!base_model_init(&impl->base, model_bytes, config)) {
        free(impl);
        return false;
    }

    model->name = "Convolutional Autoencoder (CA)";
    model->model_id = (uint32_t)(uintptr_t)impl;
    model->type_id = ModelType_model_arch_CA_m_c;
    model->instance_data = impl;
    model->run = ca_interface_run;
    model->deinit = ca_interface_deinit;
    model->update = ca_interface_update;

    return true;
}
