#ifndef MODEL_ARCH_GENERIC_H
#define MODEL_ARCH_GENERIC_H

#include "base_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GENERIC_SCORE_DIRECT = 0,      // Returns the first float value of the output tensor directly
    GENERIC_SCORE_MSE = 1          // Computes MSE between input and output tensors (autoencoder)
} generic_scoring_mode_t;

typedef struct {
    base_model_t base;
    generic_scoring_mode_t scoring_mode;
    float score_multiplier;        // Optional scale multiplier
} model_arch_generic_impl_t;

/**
 * @brief Initialize a generic TFLite model wrapper.
 * @param scoring_mode Scoring method (direct value or reconstruction MSE).
 * @param multiplier Linear scaling multiplier for output (defaults to 1.0 if <= 0.0).
 */
bool model_arch_generic_anomaly_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, generic_scoring_mode_t scoring_mode, float multiplier);

#ifdef __cplusplus
}
#endif

#endif // MODEL_ARCH_GENERIC_H
