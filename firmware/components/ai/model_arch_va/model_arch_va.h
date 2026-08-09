#ifndef MODEL_ARCH_VA_H
#define MODEL_ARCH_VA_H

#include "base_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VAE_SCORE_REC = 0,      // Reconstruction error only
    VAE_SCORE_KL,           // KL divergence only
    VAE_SCORE_COMBINED      // Weighted combination of both
} vae_scoring_mode_t;

typedef struct {
    base_model_t encoder;   // Always used. If single model, represents the full network.
    base_model_t decoder;   // Used if separate encoder/decoder are loaded.
    bool has_separate_decoder;
    
    vae_scoring_mode_t scoring_mode;
    float kl_weight;
    float rec_weight;
    float score_gamma;      // Scaling parameter for reconstruction MSE -> [0, 1]
    float score_beta;       // Scaling parameter for KL Divergence -> [0, 1]
} model_arch_va_impl_t;

/**
 * @brief Initialize the VAE model in Reconstruction scoring mode.
 */
bool model_arch_va_rec_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, float gamma);

/**
 * @brief Initialize the VAE model in KL Divergence scoring mode.
 */
bool model_arch_va_kl_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, float beta);

/**
 * @brief Initialize the VAE model in Combined scoring mode.
 */
bool model_arch_va_combined_init(anomaly_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config, float gamma, float beta, float rec_w, float kl_w);

/**
 * @brief Initialize a split VAE model (separate encoder and decoder flatbuffers).
 */
bool model_arch_va_split_init(anomaly_model_t* model, 
                              const uint8_t* enc_bytes, const base_model_config_t* enc_cfg,
                              const uint8_t* dec_bytes, const base_model_config_t* dec_cfg,
                              vae_scoring_mode_t scoring_mode, float gamma, float beta, float rec_w, float kl_w);

#ifdef __cplusplus
}
#endif

#endif // MODEL_ARCH_VA_H
