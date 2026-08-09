#ifndef ANOMALY_MODEL_INTERFACE_H
#define ANOMALY_MODEL_INTERFACE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "ensemble_types.h"

// Shared FFT data structure to avoid recalculating FFT
typedef struct {
    float* fft_data;        // Spectrogram magnitude bins
    size_t length;          // Total number of elements in fft_data (num_frames * freq_bins)
    size_t window_size;     // Window size used for STFT
    size_t hop_size;        // Hop size used for STFT
    size_t num_frames;      // Number of frames computed
    size_t freq_bins;       // Number of frequency bins (fft_size/2 + 1)
    bool is_valid;          // Flag indicating if the computed FFT data is valid
} fft_shared_data_t;

// Unified Anomaly Model Interface
typedef struct anomaly_model_s anomaly_model_t;

struct anomaly_model_s {
    const char* name;
    uint32_t model_id;
    uint8_t type_id;        // From ModelType enum
    void* instance_data;    // Implementation-specific pointer (e.g. model_arch_va_t)
    float anomaly_threshold; // Configured anomaly threshold
    
    // Unified function pointers for anomaly detection models
    // Returns true on success, and sets out_anomaly_score to [0.0 - 1.0] where 1.0 is maximum anomaly.
    bool (*run)(anomaly_model_t* model, const float* input_data, size_t input_len, const fft_shared_data_t* shared_fft, float* out_anomaly_score);
    void (*deinit)(anomaly_model_t* model);
    bool (*update)(anomaly_model_t* model, const uint8_t* new_model_bytes);
};

// Unified Router Model Interface
typedef struct router_model_s router_model_t;

struct router_model_s {
    const char* name;
    uint32_t model_id;
    uint8_t type_id;        // From ModelType enum
    void* instance_data;    // Implementation-specific pointer (e.g. model_arch_mr_t)
    size_t input_length;    // Number of data points to pass to the router
    
    // Unified function pointers for router models
    // Returns routed submodel index and route confidence score. Computes and fills out_shared_fft if requested.
    bool (*run)(router_model_t* router, const float* input_data, size_t input_len, int* out_best_route_idx, float* out_confidence, fft_shared_data_t* out_shared_fft);
    void (*deinit)(router_model_t* router);
    bool (*update)(router_model_t* router, const uint8_t* new_model_bytes);
};

#endif // ANOMALY_MODEL_INTERFACE_H
