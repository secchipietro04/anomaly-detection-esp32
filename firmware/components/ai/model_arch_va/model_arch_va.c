#include "model_arch_va.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"

static const char* TAG = "MODEL_VA";

// Forward declarations for interface functions
static bool va_interface_run(anomaly_model_t* model, const float* input_data, size_t input_len, const fft_shared_data_t* shared_fft, float* out_anomaly_score);
static void va_interface_deinit(anomaly_model_t* model);
static bool va_interface_update(anomaly_model_t* model, const uint8_t* new_model_bytes);

// Common initialization helper
static model_arch_va_impl_t* create_impl(void) {
    model_arch_va_impl_t* impl = (model_arch_va_impl_t*)calloc(1, sizeof(model_arch_va_impl_t));
    return impl;
}

static void fill_interface(anomaly_model_t* model, model_arch_va_impl_t* impl, const char* name, uint8_t type_id) {
    model->name = name;
    model->model_id = (uint32_t)(uintptr_t)impl; // Unique ID derived from pointer address
    model->type_id = type_id;
    model->instance_data = impl;
    model->run = va_interface_run;
    model->deinit = va_interface_deinit;
    model->update = va_interface_update;
}

bool model_arch_va_rec_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, float gamma) {
    if (!model || !model_bytes || !config) return false;
    model_arch_va_impl_t* impl = create_impl();
    if (!impl) return false;

    impl->scoring_mode = VAE_SCORE_REC;
    impl->score_gamma = gamma <= 0.0f ? 1.0f : gamma;
    impl->rec_weight = 1.0f;
    impl->has_separate_decoder = false;

    if (!base_model_init(&impl->encoder, model_bytes, config)) {
        free(impl);
        return false;
    }

    fill_interface(model, impl, "VAE (Reconstruction Mode)", ModelType_model_arch_VA_m_c);
    return true;
}

bool model_arch_va_kl_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, float beta) {
    if (!model || !model_bytes || !config) return false;
    model_arch_va_impl_t* impl = create_impl();
    if (!impl) return false;

    impl->scoring_mode = VAE_SCORE_KL;
    impl->score_beta = beta <= 0.0f ? 1.0f : beta;
    impl->kl_weight = 1.0f;
    impl->has_separate_decoder = false;

    if (!base_model_init(&impl->encoder, model_bytes, config)) {
        free(impl);
        return false;
    }

    fill_interface(model, impl, "VAE (KL Divergence Mode)", ModelType_model_arch_VA_m_c);
    return true;
}

bool model_arch_va_combined_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, float gamma, float beta, float rec_w, float kl_w) {
    if (!model || !model_bytes || !config) return false;
    model_arch_va_impl_t* impl = create_impl();
    if (!impl) return false;

    impl->scoring_mode = VAE_SCORE_COMBINED;
    impl->score_gamma = gamma <= 0.0f ? 1.0f : gamma;
    impl->score_beta = beta <= 0.0f ? 1.0f : beta;
    impl->rec_weight = rec_w;
    impl->kl_weight = kl_w;
    impl->has_separate_decoder = false;

    if (!base_model_init(&impl->encoder, model_bytes, config)) {
        free(impl);
        return false;
    }

    fill_interface(model, impl, "VAE (Combined Mode)", ModelType_model_arch_VA_m_c);
    return true;
}

bool model_arch_va_split_init(anomaly_model_t* model, 
                              const uint8_t* enc_bytes, const base_model_config_t* enc_cfg,
                              const uint8_t* dec_bytes, const base_model_config_t* dec_cfg,
                              vae_scoring_mode_t scoring_mode, float gamma, float beta, float rec_w, float kl_w) {
    if (!model || !enc_bytes || !enc_cfg || !dec_bytes || !dec_cfg) return false;
    model_arch_va_impl_t* impl = create_impl();
    if (!impl) return false;

    impl->scoring_mode = scoring_mode;
    impl->score_gamma = gamma <= 0.0f ? 1.0f : gamma;
    impl->score_beta = beta <= 0.0f ? 1.0f : beta;
    impl->rec_weight = rec_w;
    impl->kl_weight = kl_w;
    impl->has_separate_decoder = true;

    if (!base_model_init(&impl->encoder, enc_bytes, enc_cfg)) {
        free(impl);
        return false;
    }

    if (!base_model_init(&impl->decoder, dec_bytes, dec_cfg)) {
        base_model_deinit(&impl->encoder);
        free(impl);
        return false;
    }

    fill_interface(model, impl, "VAE (Split Mode)", ModelType_model_arch_VA_m_c);
    return true;
}

static bool va_interface_run(anomaly_model_t* model, const float* input_data, size_t input_len, const fft_shared_data_t* shared_fft, float* out_anomaly_score) {
    if (!model || !model->instance_data || !input_data || !out_anomaly_score) return false;
    model_arch_va_impl_t* impl = (model_arch_va_impl_t*)model->instance_data;

    float rec_score = 0.0f;
    float kl_score = 0.0f;

    if (!impl->has_separate_decoder) {
        // Single model: run VAE graph
        tflite_tensor_details_t* in_det = &impl->encoder.input_details;
        
        // Copy input to encoder tensor
        size_t copy_bytes = (input_len * sizeof(float) < in_det->bytes) ? input_len * sizeof(float) : in_det->bytes;
        memcpy(in_det->data, input_data, copy_bytes);

        if (!base_model_run(&impl->encoder)) {
            ESP_LOGE(TAG, "Failed to execute single-graph VAE");
            return false;
        }

        // Reconstruction computation (Raw unscaled MSE * gamma)
        if (impl->scoring_mode == VAE_SCORE_REC || impl->scoring_mode == VAE_SCORE_COMBINED) {
            tflite_tensor_details_t* out_det = &impl->encoder.output_details;
            float* rec_out = (float*)out_det->data;
            float* original = (float*)in_det->data;
            size_t elements = copy_bytes / sizeof(float);

            float sum_sq = 0.0f;
            for (size_t i = 0; i < elements; i++) {
                float diff = original[i] - rec_out[i];
                sum_sq += diff * diff;
            }
            float mse = sum_sq / (float)elements;
            rec_score = impl->score_gamma * mse;
        }

        // KL Divergence computation (Raw unscaled KL * beta)
        if (impl->scoring_mode == VAE_SCORE_KL || impl->scoring_mode == VAE_SCORE_COMBINED) {
            tflite_tensor_details_t mean_det, logvar_det;
            tflite_status_t s1 = tflite_micro_get_output_details(impl->encoder.tf_handle, 1, &mean_det);
            tflite_status_t s2 = tflite_micro_get_output_details(impl->encoder.tf_handle, 2, &logvar_det);

            if (s1 == TFLITE_WRAPPER_OK && s2 == TFLITE_WRAPPER_OK) {
                float* mean = (float*)mean_det.data;
                float* logvar = (float*)logvar_det.data;
                size_t lat_elems = mean_det.bytes / sizeof(float);

                float kl_div = 0.0f;
                for (size_t i = 0; i < lat_elems; i++) {
                    kl_div += -0.5f * (1.0f + logvar[i] - mean[i] * mean[i] - expf(logvar[i]));
                }
                kl_score = impl->score_beta * kl_div;
            } else {
                kl_score = 0.0f;
            }
        }
    } else {
        // Split model VAE: Encoder first, then decoder
        tflite_tensor_details_t* enc_in = &impl->encoder.input_details;
        size_t copy_bytes = (input_len * sizeof(float) < enc_in->bytes) ? input_len * sizeof(float) : enc_in->bytes;
        memcpy(enc_in->data, input_data, copy_bytes);

        if (!base_model_run(&impl->encoder)) {
            ESP_LOGE(TAG, "Failed to run VAE encoder");
            return false;
        }

        tflite_tensor_details_t mean_det, logvar_det;
        tflite_micro_get_output_details(impl->encoder.tf_handle, 0, &mean_det);
        tflite_micro_get_output_details(impl->encoder.tf_handle, 1, &logvar_det);

        float* mean = (float*)mean_det.data;
        float* logvar = (float*)logvar_det.data;
        size_t latent_dim = mean_det.bytes / sizeof(float);

        // KL Divergence
        if (impl->scoring_mode == VAE_SCORE_KL || impl->scoring_mode == VAE_SCORE_COMBINED) {
            float kl_div = 0.0f;
            for (size_t i = 0; i < latent_dim; i++) {
                kl_div += -0.5f * (1.0f + logvar[i] - mean[i] * mean[i] - expf(logvar[i]));
            }
            kl_score = impl->score_beta * kl_div;
        }

        // Pass Mean directly to Decoder
        if (impl->scoring_mode == VAE_SCORE_REC || impl->scoring_mode == VAE_SCORE_COMBINED) {
            tflite_tensor_details_t* dec_in = &impl->decoder.input_details;
            size_t dec_copy_bytes = (mean_det.bytes < dec_in->bytes) ? mean_det.bytes : dec_in->bytes;
            memcpy(dec_in->data, mean, dec_copy_bytes);

            if (!base_model_run(&impl->decoder)) {
                ESP_LOGE(TAG, "Failed to run VAE decoder");
                return false;
            }

            tflite_tensor_details_t* dec_out = &impl->decoder.output_details;
            float* rec_out = (float*)dec_out->data;
            float* original = (float*)enc_in->data;
            size_t elements = copy_bytes / sizeof(float);

            float sum_sq = 0.0f;
            for (size_t i = 0; i < elements; i++) {
                float diff = original[i] - rec_out[i];
                sum_sq += diff * diff;
            }
            float mse = sum_sq / (float)elements;
            rec_score = impl->score_gamma * mse;
        }
    }

    // Combine scores based on weights (Raw values)
    if (impl->scoring_mode == VAE_SCORE_REC) {
        *out_anomaly_score = rec_score;
    } else if (impl->scoring_mode == VAE_SCORE_KL) {
        *out_anomaly_score = kl_score;
    } else {
        float sum_w = impl->rec_weight + impl->kl_weight;
        if (sum_w > 0.0f) {
            *out_anomaly_score = (impl->rec_weight * rec_score + impl->kl_weight * kl_score) / sum_w;
        } else {
            *out_anomaly_score = 0.0f;
        }
    }

    return true;
}

static void va_interface_deinit(anomaly_model_t* model) {
    if (!model || !model->instance_data) return;
    model_arch_va_impl_t* impl = (model_arch_va_impl_t*)model->instance_data;
    base_model_deinit(&impl->encoder);
    if (impl->has_separate_decoder) {
        base_model_deinit(&impl->decoder);
    }
    free(impl);
    model->instance_data = NULL;
}

static bool va_interface_update(anomaly_model_t* model, const uint8_t* new_model_bytes) {
    if (!model || !model->instance_data || !new_model_bytes) return false;
    model_arch_va_impl_t* impl = (model_arch_va_impl_t*)model->instance_data;
    return base_model_reload(&impl->encoder, new_model_bytes);
}
