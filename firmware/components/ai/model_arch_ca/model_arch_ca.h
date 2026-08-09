#ifndef MODEL_ARCH_CA_H
#define MODEL_ARCH_CA_H

#include "base_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    base_model_t base;
    float score_gamma;      // Scaling parameter for reconstruction MSE -> [0, 1]
} model_arch_ca_impl_t;

/**
 * @brief Initialize the Convolutional Autoencoder (CA) anomaly detection model.
 * @param gamma Scaling parameter for mapping reconstruction MSE to anomaly score.
 */
bool model_arch_ca_anomaly_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, float gamma);

#ifdef __cplusplus
}
#endif

#endif // MODEL_ARCH_CA_H
