#ifndef MODEL_ARCH_TN_H
#define MODEL_ARCH_TN_H

#include "base_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    base_model_t base;
    size_t window_size;
    size_t hop_size;
    size_t fft_size;
    float score_gamma;      // Scaling parameter for reconstruction MSE -> [0, 1]
} model_arch_tn_impl_t;

/**
 * @brief Initialize the TimesNet (TN) anomaly detection model.
 * @param window_size STFT window size (must be power of 2).
 * @param hop_size STFT hop size.
 * @param gamma Scaling parameter for mapping reconstruction MSE to anomaly score.
 */
bool model_arch_tn_anomaly_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, size_t window_size, size_t hop_size, float gamma);

#ifdef __cplusplus
}
#endif

#endif // MODEL_ARCH_TN_H
