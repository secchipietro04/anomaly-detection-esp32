#ifndef MODEL_ENSEMBLE_H
#define MODEL_ENSEMBLE_H

#include "anomaly_model_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_SUBMODELS 16

typedef struct {
    router_model_t* router;
    anomaly_model_t* submodels[MAX_SUBMODELS]; // Direct mapping: slot index corresponds to router output index
    float router_confidence_threshold;         // Minimum confidence threshold for routing
} model_ensemble_t;

/**
 * @brief Initialize the model ensemble.
 * @param confidence_threshold Minimum router confidence required.
 */
bool model_ensemble_init(model_ensemble_t* ensemble, float confidence_threshold);

/**
 * @brief Deinitialize the model ensemble (unloads everything, resets structure).
 */
void model_ensemble_deinit(model_ensemble_t* ensemble);

/**
 * @brief Set or replace the router model in the ensemble.
 */
bool model_ensemble_set_router(model_ensemble_t* ensemble, router_model_t* router);

/**
 * @brief Phase 1: Execute the router model to classify the segment.
 * @param input_series Raw input time series signal segment.
 * @param input_len Length of input series.
 * @param out_best_route_idx Pointer to output index representing the most likely model.
 * @param out_confidence Pointer to output router confidence.
 * @param out_shared_fft Pointer to output FFT structure. If out_shared_fft->fft_data is pre-allocated by the caller,
 *                       the router will write the STFT directly into it.
 * @return true on success, false on failure.
 */
bool model_ensemble_run_router(model_ensemble_t* ensemble, const float* input_series, size_t input_len, int* out_best_route_idx, float* out_confidence, fft_shared_data_t* out_shared_fft);

/**
 * @brief Load a single anomaly detection submodel into a specific slot.
 *        Allows swapping models dynamically to manage RAM usage.
 * @param submodel Pointer to the submodel instance to load.
 * @param route_idx The slot index corresponding to the router class (0 to MAX_SUBMODELS-1).
 * @param anomaly_threshold Anomaly threshold configured for this model. Stored directly inside submodel.
 */
bool model_ensemble_load_submodel(model_ensemble_t* ensemble, anomaly_model_t* submodel, int route_idx, float anomaly_threshold);

/**
 * @brief Unload a submodel from the ensemble to free RAM.
 * @param model_id The unique model ID of the submodel to unload.
 */
bool model_ensemble_unload_submodel(model_ensemble_t* ensemble, uint32_t model_id);

/**
 * @brief Phase 2: Execute an anomaly detection submodel by its unique ID.
 * @param model_id The unique ID of the model to execute.
 * @param input_series Raw segment data.
 * @param input_len Length of raw segment.
 * @param shared_fft Pointer to the precalculated FFT data struct. Can be NULL or marked invalid if no FFT is available.
 * @param out_anomaly_score Pointer to output raw anomaly score.
 * @param out_is_anomaly Pointer to output boolean (true if raw score >= model's own anomaly threshold).
 * @return true if the model was found and run successfully, false if the model is not loaded or failed.
 */
bool model_ensemble_run_submodel(model_ensemble_t* ensemble, uint32_t model_id, const float* input_series, size_t input_len, const fft_shared_data_t* shared_fft, float* out_anomaly_score, bool* out_is_anomaly);

#ifdef __cplusplus
}
#endif

#endif // MODEL_ENSEMBLE_H
