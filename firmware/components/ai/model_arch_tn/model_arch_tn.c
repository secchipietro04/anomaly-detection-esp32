#include "model_arch_tn.h"
#include "esp_fft_wrapper.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"

static const char* TAG = "MODEL_TN";

static bool tn_interface_run(anomaly_model_t* model, const float* input_data, size_t input_len, const fft_shared_data_t* shared_fft, float* out_anomaly_score) {
    if (!model || !model->instance_data || !input_data || !out_anomaly_score) return false;
    model_arch_tn_impl_t* impl = (model_arch_tn_impl_t*)model->instance_data;

    tflite_tensor_details_t* in_det = &impl->base.input_details;
    tflite_tensor_details_t* out_det = &impl->base.output_details;

    size_t freq_bins = impl->fft_size / 2 + 1;
    size_t num_frames = (input_len - impl->window_size) / impl->hop_size + 1;
    size_t required_bytes = num_frames * freq_bins * sizeof(float);

    if (required_bytes > in_det->bytes) {
        ESP_LOGE(TAG, "STFT output size (%zu) exceeds TimesNet input tensor capacity (%zu)", required_bytes, in_det->bytes);
        return false;
    }

    // Zero-copy sharing: check if we can reuse computed STFT
    if (shared_fft && shared_fft->is_valid &&
        shared_fft->window_size == impl->window_size &&
        shared_fft->hop_size == impl->hop_size &&
        shared_fft->freq_bins == freq_bins &&
        shared_fft->num_frames == num_frames &&
        shared_fft->fft_data != NULL) {
        
        memcpy(in_det->data, shared_fft->fft_data, required_bytes);
        ESP_LOGD(TAG, "Reusing precalculated STFT spectrogram data in TimesNet");
    } else {
        if (!esp_fft_wrapper_stft(input_data, input_len, impl->window_size, impl->hop_size, impl->fft_size, (float*)in_det->data)) {
            ESP_LOGE(TAG, "Failed to compute STFT locally for TimesNet");
            return false;
        }
    }

    if (!base_model_run(&impl->base)) {
        ESP_LOGE(TAG, "Failed to run TimesNet model");
        return false;
    }

    // Compute raw reconstruction MSE
    float* rec_out = (float*)out_det->data;
    float* original = (float*)in_det->data;
    size_t elements = required_bytes / sizeof(float);

    float sum_sq = 0.0f;
    for (size_t i = 0; i < elements; i++) {
        float diff = original[i] - rec_out[i];
        sum_sq += diff * diff;
    }
    float mse = sum_sq / (float)elements;

    *out_anomaly_score = impl->score_gamma * mse;
    return true;
}

static void tn_interface_deinit(anomaly_model_t* model) {
    if (!model || !model->instance_data) return;
    model_arch_tn_impl_t* impl = (model_arch_tn_impl_t*)model->instance_data;
    base_model_deinit(&impl->base);
    free(impl);
    model->instance_data = NULL;
}

static bool tn_interface_update(anomaly_model_t* model, const uint8_t* new_model_bytes) {
    if (!model || !model->instance_data || !new_model_bytes) return false;
    model_arch_tn_impl_t* impl = (model_arch_tn_impl_t*)model->instance_data;
    return base_model_reload(&impl->base, new_model_bytes);
}

bool model_arch_tn_anomaly_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, size_t window_size, size_t hop_size, float gamma) {
    if (!model || !model_bytes || !config || window_size == 0 || hop_size == 0) return false;
    model_arch_tn_impl_t* impl = (model_arch_tn_impl_t*)calloc(1, sizeof(model_arch_tn_impl_t));
    if (!impl) return false;

    impl->window_size = window_size;
    impl->hop_size = hop_size;
    impl->score_gamma = gamma <= 0.0f ? 1.0f : gamma;

    impl->fft_size = 1;
    while (impl->fft_size < window_size) {
        impl->fft_size <<= 1;
    }

    if (!base_model_init(&impl->base, model_bytes, config)) {
        free(impl);
        return false;
    }

    model->name = "TimesNet Model (rt mode)";
    model->model_id = (uint32_t)(uintptr_t)impl;
    model->type_id = ModelType_model_arch_TN_m_c;
    model->instance_data = impl;
    model->run = tn_interface_run;
    model->deinit = tn_interface_deinit;
    model->update = tn_interface_update;

    return true;
}
