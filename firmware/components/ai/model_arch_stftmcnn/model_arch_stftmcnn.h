#ifndef MODEL_ARCH_STFTMCNN_H
#define MODEL_ARCH_STFTMCNN_H

#include "base_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    base_model_t base;
    size_t window_size;
    size_t hop_size;
    size_t fft_size;
    float score_gamma;              // Scaling parameter for anomaly mode
    
    // Persistent spectrogram buffer to support zero-copy STFT sharing
    float* spectrogram_buffer;
    size_t spectrogram_len;
} model_arch_stftmcnn_impl_t;

/**
 * @brief Initialize STFT + Micro-CNN as an Anomaly Detection Model.
 */
bool model_arch_stftmcnn_anomaly_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, size_t window_size, size_t hop_size, float gamma);

/**
 * @brief Initialize STFT + Micro-CNN as a Router Model.
 */
bool model_arch_stftmcnn_router_init(router_model_t* router, const uint8_t* model_bytes, const base_model_config_t* config, size_t window_size, size_t hop_size, size_t input_length);

#ifdef __cplusplus
}
#endif

#endif // MODEL_ARCH_STFTMCNN_H
