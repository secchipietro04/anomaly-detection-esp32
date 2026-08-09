#ifndef MODEL_ARCH_DENSE_H
#define MODEL_ARCH_DENSE_H

#include "base_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    base_model_t base;
    float score_gamma;      // Scaling parameter for reconstruction MSE -> [0, 1]
} model_arch_dense_impl_t;

/**
 * @brief Initialize the Dense Model as an Anomaly Detection Model (DA).
 * @param gamma Scaling parameter for reconstruction scoring.
 */
bool model_arch_dense_anomaly_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, float gamma);

/**
 * @brief Initialize the Dense Model as a Router Model.
 * @param input_length Total input dimensions expected by the router.
 */
bool model_arch_dense_router_init(router_model_t* router, const uint8_t* model_bytes, const base_model_config_t* config, size_t input_length);

#ifdef __cplusplus
}
#endif

#endif // MODEL_ARCH_DENSE_H
