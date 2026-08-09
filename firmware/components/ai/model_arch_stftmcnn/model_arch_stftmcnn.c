#include "model_arch_stftmcnn.h"
#include "esp_fft_wrapper.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"

static const char* TAG = "MODEL_STFTMCNN";

// Helper function to calculate STFT and populate the persistent spectrogram buffer
static bool compute_stft_spectrogram(model_arch_stftmcnn_impl_t* impl, const float* input_series, size_t input_len, size_t* out_num_frames, size_t* out_freq_bins) {
    size_t w_size = impl->window_size;
    size_t h_size = impl->hop_size;
    size_t fft_len = impl->fft_size;
    size_t freq_bins = fft_len / 2 + 1;

    if (input_len < w_size) {
        ESP_LOGE(TAG, "Input series length (%zu) is shorter than window size (%zu)", input_len, w_size);
        return false;
    }

    size_t num_frames = (input_len - w_size) / h_size + 1;
    size_t required_elements = num_frames * freq_bins;

    if (impl->spectrogram_len < required_elements || !impl->spectrogram_buffer) {
        float* new_buf = (float*)realloc(impl->spectrogram_buffer, required_elements * sizeof(float));
        if (!new_buf) {
            ESP_LOGE(TAG, "Failed to allocate spectrogram buffer (%zu elements)", required_elements);
            return false;
        }
        impl->spectrogram_buffer = new_buf;
        impl->spectrogram_len = required_elements;
    }

    if (!esp_fft_wrapper_stft(input_series, input_len, w_size, h_size, fft_len, impl->spectrogram_buffer)) {
        ESP_LOGE(TAG, "STFT computation failed");
        return false;
    }

    *out_num_frames = num_frames;
    *out_freq_bins = freq_bins;
    return true;
}

// ==========================================
// Anomaly Detection Interface Implementation
// ==========================================
static bool stftmcnn_anomaly_run(anomaly_model_t* model, const float* input_data, size_t input_len, const fft_shared_data_t* shared_fft, float* out_anomaly_score) {
    if (!model || !model->instance_data || !input_data || !out_anomaly_score) return false;
    model_arch_stftmcnn_impl_t* impl = (model_arch_stftmcnn_impl_t*)model->instance_data;

    tflite_tensor_details_t* in_det = &impl->base.input_details;
    tflite_tensor_details_t* out_det = &impl->base.output_details;

    size_t num_frames = 0, freq_bins = 0;
    
    // Zero-copy check: can we reuse a precalculated STFT spectrogram?
    size_t expected_freq_bins = impl->fft_size / 2 + 1;
    size_t expected_num_frames = (input_len - impl->window_size) / impl->hop_size + 1;
    size_t required_bytes = expected_num_frames * expected_freq_bins * sizeof(float);

    if (required_bytes > in_det->bytes) {
        ESP_LOGE(TAG, "Spectrogram size (%zu) exceeds input tensor capacity (%zu)", required_bytes, in_det->bytes);
        return false;
    }

    if (shared_fft && shared_fft->is_valid &&
        shared_fft->window_size == impl->window_size &&
        shared_fft->hop_size == impl->hop_size &&
        shared_fft->freq_bins == expected_freq_bins &&
        shared_fft->num_frames == expected_num_frames &&
        shared_fft->fft_data != NULL) {
        
        memcpy(in_det->data, shared_fft->fft_data, required_bytes);
        ESP_LOGD(TAG, "Reusing precalculated STFT spectrogram");
    } else {
        if (!compute_stft_spectrogram(impl, input_data, input_len, &num_frames, &freq_bins)) {
            return false;
        }
        memcpy(in_det->data, impl->spectrogram_buffer, required_bytes);
    }

    if (!base_model_run(&impl->base)) {
        ESP_LOGE(TAG, "Failed to execute STFTmCNN Anomaly Model");
        return false;
    }

    // Compute raw reconstruction MSE of the spectrogram image
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

static void stftmcnn_anomaly_deinit(anomaly_model_t* model) {
    if (!model || !model->instance_data) return;
    model_arch_stftmcnn_impl_t* impl = (model_arch_stftmcnn_impl_t*)model->instance_data;
    base_model_deinit(&impl->base);
    if (impl->spectrogram_buffer) {
        free(impl->spectrogram_buffer);
    }
    free(impl);
    model->instance_data = NULL;
}

static bool stftmcnn_anomaly_update(anomaly_model_t* model, const uint8_t* new_model_bytes) {
    if (!model || !model->instance_data || !new_model_bytes) return false;
    model_arch_stftmcnn_impl_t* impl = (model_arch_stftmcnn_impl_t*)model->instance_data;
    return base_model_reload(&impl->base, new_model_bytes);
}

bool model_arch_stftmcnn_anomaly_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, size_t window_size, size_t hop_size, float gamma) {
    if (!model || !model_bytes || !config || window_size == 0 || hop_size == 0) return false;
    model_arch_stftmcnn_impl_t* impl = (model_arch_stftmcnn_impl_t*)calloc(1, sizeof(model_arch_stftmcnn_impl_t));
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

    model->name = "STFT + Micro-CNN Anomaly Detection Model";
    model->model_id = (uint32_t)(uintptr_t)impl;
    model->type_id = ModelType_model_router_STFTmCNN_m_c;
    model->instance_data = impl;
    model->run = stftmcnn_anomaly_run;
    model->deinit = stftmcnn_anomaly_deinit;
    model->update = stftmcnn_anomaly_update;

    return true;
}

// ==========================================
// Router Interface Implementation
// ==========================================
static bool stftmcnn_router_run(router_model_t* router, const float* input_data, size_t input_len, int* out_best_route_idx, float* out_confidence, fft_shared_data_t* out_shared_fft) {
    if (!router || !router->instance_data || !input_data || !out_best_route_idx || !out_confidence) return false;
    model_arch_stftmcnn_impl_t* impl = (model_arch_stftmcnn_impl_t*)router->instance_data;

    tflite_tensor_details_t* in_det = &impl->base.input_details;
    tflite_tensor_details_t* out_det = &impl->base.output_details;

    size_t w_size = impl->window_size;
    size_t h_size = impl->hop_size;
    size_t fft_len = impl->fft_size;
    size_t freq_bins = fft_len / 2 + 1;
    size_t num_frames = (input_len - w_size) / h_size + 1;
    size_t required_elements = num_frames * freq_bins;
    size_t required_bytes = required_elements * sizeof(float);

    if (required_bytes > in_det->bytes) {
        ESP_LOGE(TAG, "Spectrogram size (%zu) exceeds router input tensor capacity (%zu)", required_bytes, in_det->bytes);
        return false;
    }

    float* target_fft_buf = NULL;
    bool using_caller_buf = false;

    if (out_shared_fft && out_shared_fft->fft_data && out_shared_fft->length >= required_elements) {
        target_fft_buf = out_shared_fft->fft_data;
        using_caller_buf = true;
    } else {
        // Fall back to internal spectrogram buffer
        if (impl->spectrogram_len < required_elements || !impl->spectrogram_buffer) {
            float* new_buf = (float*)realloc(impl->spectrogram_buffer, required_elements * sizeof(float));
            if (!new_buf) {
                ESP_LOGE(TAG, "Failed to allocate internal spectrogram buffer");
                return false;
            }
            impl->spectrogram_buffer = new_buf;
            impl->spectrogram_len = required_elements;
        }
        target_fft_buf = impl->spectrogram_buffer;
    }

    // Compute STFT directly into target buffer (caller-allocated or internal)
    if (!esp_fft_wrapper_stft(input_data, input_len, w_size, h_size, fft_len, target_fft_buf)) {
        ESP_LOGE(TAG, "STFT computation failed");
        return false;
    }

    // Load computed STFT spectrogram into input tensor
    memcpy(in_det->data, target_fft_buf, required_bytes);

    if (!base_model_run(&impl->base)) {
        ESP_LOGE(TAG, "Failed to execute STFTmCNN Router");
        return false;
    }

    // Get routed class index & confidence from output tensor
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

    // Export spectrogram metadata
    if (out_shared_fft) {
        if (!using_caller_buf) {
            out_shared_fft->fft_data = impl->spectrogram_buffer;
            out_shared_fft->length = impl->spectrogram_len;
        }
        out_shared_fft->window_size = w_size;
        out_shared_fft->hop_size = h_size;
        out_shared_fft->num_frames = num_frames;
        out_shared_fft->freq_bins = freq_bins;
        out_shared_fft->is_valid = true;
    }

    return true;
}

static void stftmcnn_router_deinit(router_model_t* router) {
    if (!router || !router->instance_data) return;
    model_arch_stftmcnn_impl_t* impl = (model_arch_stftmcnn_impl_t*)router->instance_data;
    base_model_deinit(&impl->base);
    if (impl->spectrogram_buffer) {
        free(impl->spectrogram_buffer);
    }
    free(impl);
    router->instance_data = NULL;
}

static bool stftmcnn_router_update(router_model_t* router, const uint8_t* new_model_bytes) {
    if (!router || !router->instance_data || !new_model_bytes) return false;
    model_arch_stftmcnn_impl_t* impl = (model_arch_stftmcnn_impl_t*)router->instance_data;
    return base_model_reload(&impl->base, new_model_bytes);
}

bool model_arch_stftmcnn_router_init(router_model_t* router, const uint8_t* model_bytes, const base_model_config_t* config, size_t window_size, size_t hop_size, size_t input_length) {
    if (!router || !model_bytes || !config || window_size == 0 || hop_size == 0) return false;
    model_arch_stftmcnn_impl_t* impl = (model_arch_stftmcnn_impl_t*)calloc(1, sizeof(model_arch_stftmcnn_impl_t));
    if (!impl) return false;

    impl->window_size = window_size;
    impl->hop_size = hop_size;

    impl->fft_size = 1;
    while (impl->fft_size < window_size) {
        impl->fft_size <<= 1;
    }

    if (!base_model_init(&impl->base, model_bytes, config)) {
        free(impl);
        return false;
    }

    router->name = "STFT + Micro-CNN Router";
    router->model_id = (uint32_t)(uintptr_t)impl;
    router->type_id = ModelType_model_router_STFTmCNN_m_c;
    router->instance_data = impl;
    router->input_length = input_length;
    router->run = stftmcnn_router_run;
    router->deinit = stftmcnn_router_deinit;
    router->update = stftmcnn_router_update;

    return true;
}
