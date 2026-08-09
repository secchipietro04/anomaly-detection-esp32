#ifndef MODEL_ARCH_MR_H
#define MODEL_ARCH_MR_H

#include "base_model.h"

#ifdef __cplusplus
extern "C" {
#endif

// MiniRocket configuration parameters for custom C inference
typedef struct {
    size_t num_kernels;       // Number of random kernels (default 84)
    size_t num_features;      // Total number of features (e.g. 999 or num_kernels * num_biases)
    const int8_t* kernels;    // Pointer to kernel weights (num_kernels x 9)
    const int32_t* dilations; // Pointer to dilations for each feature
    const float* biases;      // Pointer to biases for each feature
    const float* weights;     // Pointer to linear classifier weights (num_classes x num_features)
    const float* intercepts;  // Pointer to linear classifier intercepts (num_classes)
    size_t num_classes;       // Number of output classes for model decision
    size_t input_length;      // Length of input time series
    size_t anomaly_class_idx; // Index of class representing anomaly in classifier (default 1)
} model_arch_mr_config_t;

typedef struct {
    base_model_t base;        // Embedded TFLite base model (used if tflite mode is active)
    bool use_tflite;          // Flag indicating if model is running via TFLite Micro interpreter
    model_arch_mr_config_t config;
    float* feature_buffer;    // Temporary buffer to hold computed PPV features for custom C mode
} model_arch_mr_impl_t;

/**
 * @brief Initialize MiniRocket as an Anomaly Detection Model.
 */
bool model_arch_mr_anomaly_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, const model_arch_mr_config_t* mr_cfg);

/**
 * @brief Initialize MiniRocket as a Router Model.
 */
bool model_arch_mr_router_init(router_model_t* router, const uint8_t* model_bytes, const base_model_config_t* config, const model_arch_mr_config_t* mr_cfg);

#ifdef __cplusplus
}
#endif

#endif // MODEL_ARCH_MR_H
