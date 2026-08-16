#ifndef MODEL_ENSEMBLE_H
#define MODEL_ENSEMBLE_H

#include "base_model.h"
#include "ring_buffer.h"
#include "model_cache.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_ROUTES 16

#define DEFAULT_ROUTER_ARENA_SIZE   (64 * 1024)
#define DEFAULT_MEMORY_ARENA_SIZE   (64 * 1024)
#define DEFAULT_SUBMODEL_ARENA_SIZE  (128 * 1024)

#define DEFAULT_SUBMODEL_CACHE_CAPACITY 4

typedef struct {
    uint32_t out_idx;
    uint32_t m_id;
} route_entry_t;

typedef struct {
    uint32_t raw_bins;            // Input FFT width (default 256)
    uint32_t history_depth;        // Shared ring buffer size (N slices, default 16)
    uint32_t warmup_steps;         // Warmup period before alerts/scoring
    uint32_t warmup_steps_done;    // Uptime steps processed

    // Shared circular ring buffer wrapper
    ring_buffer_t ring_buffer;

    // State dimension configuration
    uint32_t state_dim;            // Size of h_state and c_state recurrent vectors

    // Static memory buffers for Archetype 3 (Memory Backbone)
    float* h_state;
    float* c_state;

    // Active resident models in RAM
    ModelInstance_t router_model;
    bool router_model_loaded;

    ModelInstance_t memory_model;
    bool memory_model_loaded;

    // LIFO cache pool for active submodels
    model_cache_t submodel_cache;

    // Submodel routing table
    route_entry_t routes[MAX_ROUTES];
    uint32_t num_routes;

    // Arena sizes configuration
    size_t router_arena_size;
    size_t memory_arena_size;
    size_t submodel_arena_size;
} model_ensemble_t;

/**
 * @brief Initialize the model ensemble configuration and allocate buffers.
 * @param ensemble Pointer to the model ensemble.
 * @param raw_bins Dimension of the incoming FFT magnitudes.
 * @param history_depth Number of slices in the shared ring buffer.
 * @param warmup_steps Active warmup steps before inference is evaluated.
 * @return true on success, false on allocation failure.
 */
bool model_ensemble_init(model_ensemble_t* ensemble, uint32_t raw_bins, uint32_t history_depth, uint32_t warmup_steps);

/**
 * @brief Deinitialize the model ensemble and free all internal resources.
 * @param ensemble Pointer to the model ensemble.
 */
void model_ensemble_deinit(model_ensemble_t* ensemble);

/**
 * @brief Resets the circular buffer, history of model IDs, and zeroes static memory recurrent state buffers.
 * @param ensemble Pointer to the model ensemble.
 */
void ensemble_reset_state(model_ensemble_t* ensemble);

/**
 * @brief Loads the resident router model into RAM.
 * @param ensemble Pointer to the model ensemble.
 * @param model_data Flatbuffer model binary byte array.
 * @param model_size Size of the model binary in bytes.
 * @return true on success, false on failure.
 */
bool model_ensemble_load_router(model_ensemble_t* ensemble, const uint8_t* model_data, size_t model_size);

/**
 * @brief Loads the resident memory backbone model into RAM.
 * @param ensemble Pointer to the model ensemble.
 * @param model_data Flatbuffer model binary byte array.
 * @param model_size Size of the model binary in bytes.
 * @return true on success, false on failure.
 */
bool model_ensemble_load_memory(model_ensemble_t* ensemble, const uint8_t* model_data, size_t model_size);

/**
 * @brief Configure routing mapping tables.
 * @param ensemble Pointer to the model ensemble.
 * @param routes Pointer to route mapping array.
 * @param num_routes Number of active routes.
 */
void model_ensemble_set_routes(model_ensemble_t* ensemble, const route_entry_t* routes, uint32_t num_routes);

/**
 * @brief Loads a submodel into the cache pool.
 * @param ensemble Pointer to the model ensemble.
 * @param model_data Flatbuffer model binary byte array.
 * @param model_size Size of the model binary in bytes.
 * @return MODEL_SUCCESS on success, negative error code on failure (e.g. MODEL_ERR_NO_MEM).
 */
int model_ensemble_load_submodel(model_ensemble_t* ensemble, const uint8_t* model_data, size_t model_size);

/**
 * @brief Configure the maximum number of submodels kept resident in RAM.
 * @param ensemble Pointer to the model ensemble.
 * @param capacity Number of cached submodel slots.
 */
void model_ensemble_set_cache_capacity(model_ensemble_t* ensemble, uint32_t capacity);

/**
 * @brief Ingests an FFT slice, updates memory, and determines the routed submodel index.
 * @param ensemble Pointer to the model ensemble.
 * @param fft_raw Pointer to raw FFT slice (size raw_bins).
 * @param out_target_submodel_id Pointer to output routed submodel ID.
 * @param out_already_loaded Pointer to output boolean (true if submodel is currently cached in RAM).
 * @param out_warmup_more_data Pointer to output boolean (true if still warming up).
 * @return true on success, false on failure.
 */
bool model_ensemble_step_1(model_ensemble_t* ensemble, const float* fft_raw, uint32_t* out_target_submodel_id, bool* out_already_loaded, bool* out_warmup_more_data);

/**
 * @brief Executes the active submodel (Archetype 1) and computes the anomaly score.
 * @param ensemble Pointer to the model ensemble.
 * @param out_anomaly_score Pointer to output anomaly score.
 * @param out_is_anomaly Pointer to output alert flag.
 * @return true on success, false on failure (e.g. no submodel loaded).
 */
bool model_ensemble_step_2(model_ensemble_t* ensemble, float* out_anomaly_score, bool* out_is_anomaly);

#ifdef __cplusplus
}
#endif

#endif // MODEL_ENSEMBLE_H
