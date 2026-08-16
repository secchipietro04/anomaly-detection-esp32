#include "model_ensemble.h"
#include "dsp_utils.h"
#include "loss_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"

static const char* TAG = "MODEL_ENSEMBLE";

bool model_ensemble_init(model_ensemble_t* ensemble, uint32_t raw_bins, uint32_t history_depth, uint32_t warmup_steps) {
    if (!ensemble) return false;
    memset(ensemble, 0, sizeof(model_ensemble_t));

    ensemble->raw_bins = raw_bins;
    ensemble->history_depth = history_depth;
    ensemble->warmup_steps = warmup_steps;

    if (!ring_buffer_init(&ensemble->ring_buffer, history_depth, raw_bins)) {
        return false;
    }

    ensemble->router_arena_size = 64 * 1024;
    ensemble->memory_arena_size = 64 * 1024;
    ensemble->submodel_arena_size = 128 * 1024;

    model_cache_init(&ensemble->submodel_cache, DEFAULT_SUBMODEL_CACHE_CAPACITY, ensemble->submodel_arena_size);

    return true;
}

void model_ensemble_deinit(model_ensemble_t* ensemble) {
    if (!ensemble) return;

    if (ensemble->router_model_loaded) {
        model_instance_deinit(&ensemble->router_model);
        ensemble->router_model_loaded = false;
    }
    if (ensemble->memory_model_loaded) {
        model_instance_deinit(&ensemble->memory_model);
        ensemble->memory_model_loaded = false;
    }

    model_cache_clear(&ensemble->submodel_cache);
    ring_buffer_free(&ensemble->ring_buffer);

    if (ensemble->h_state) {
        free(ensemble->h_state);
        ensemble->h_state = NULL;
    }
    if (ensemble->c_state) {
        free(ensemble->c_state);
        ensemble->c_state = NULL;
    }
}

void ensemble_reset_state(model_ensemble_t* ensemble) {
    if (!ensemble) return;

    ring_buffer_clear(&ensemble->ring_buffer);

    if (ensemble->h_state && ensemble->state_dim > 0) {
        memset(ensemble->h_state, 0, ensemble->state_dim * sizeof(float));
    }
    if (ensemble->c_state && ensemble->state_dim > 0) {
        memset(ensemble->c_state, 0, ensemble->state_dim * sizeof(float));
    }

    ensemble->warmup_steps_done = 0;
    ESP_LOGI(TAG, "Ensemble execution state reset");
}

bool model_ensemble_load_router(model_ensemble_t* ensemble, const uint8_t* model_data, size_t model_size) {
    if (!ensemble || !model_data || model_size == 0) return false;

    if (ensemble->router_model_loaded) {
        model_instance_deinit(&ensemble->router_model);
        ensemble->router_model_loaded = false;
    }

    int ret = model_instance_init(&ensemble->router_model, model_data, model_size, ensemble->router_arena_size);
    if (ret != MODEL_SUCCESS) {
        ESP_LOGE(TAG, "Failed to load router model instance");
        return false;
    }

    ensemble->router_model_loaded = true;
    return true;
}

bool model_ensemble_load_memory(model_ensemble_t* ensemble, const uint8_t* model_data, size_t model_size) {
    if (!ensemble || !model_data || model_size == 0) return false;

    if (ensemble->memory_model_loaded) {
        model_instance_deinit(&ensemble->memory_model);
        ensemble->memory_model_loaded = false;
    }

    int ret = model_instance_init(&ensemble->memory_model, model_data, model_size, ensemble->memory_arena_size);
    if (ret != MODEL_SUCCESS) {
        ESP_LOGE(TAG, "Failed to load memory backbone model instance");
        return false;
    }

    ensemble->memory_model_loaded = true;
    ensemble->state_dim = ensemble->memory_model.config.d;

    if (ensemble->h_state) free(ensemble->h_state);
    if (ensemble->c_state) free(ensemble->c_state);

    ensemble->h_state = (float*)calloc(ensemble->state_dim, sizeof(float));
    ensemble->c_state = (float*)calloc(ensemble->state_dim, sizeof(float));

    if (!ensemble->h_state || !ensemble->c_state) {
        model_instance_deinit(&ensemble->memory_model);
        ensemble->memory_model_loaded = false;
        return false;
    }

    return true;
}

void model_ensemble_set_routes(model_ensemble_t* ensemble, const route_entry_t* routes, uint32_t num_routes) {
    if (!ensemble || !routes) return;

    ensemble->num_routes = num_routes < MAX_ROUTES ? num_routes : MAX_ROUTES;
    for (uint32_t i = 0; i < ensemble->num_routes; i++) {
        ensemble->routes[i] = routes[i];
    }
}

int model_ensemble_load_submodel(model_ensemble_t* ensemble, const uint8_t* model_data, size_t model_size) {
    if (!ensemble || !model_data || model_size == 0) return MODEL_ERR_GENERIC;

    // Delegate to LIFO model cache insert (evicts least recently used if cache limit reached)
    return model_cache_insert(&ensemble->submodel_cache, model_data, model_size);
}

void model_ensemble_set_cache_capacity(model_ensemble_t* ensemble, uint32_t capacity) {
    if (!ensemble) return;
    model_cache_set_capacity(&ensemble->submodel_cache, capacity);
}

bool model_ensemble_step_1(model_ensemble_t* ensemble, const float* fft_raw, uint32_t* out_target_submodel_id, bool* out_already_loaded, bool* out_warmup_more_data) {
    if (!ensemble || !fft_raw || !out_target_submodel_id || !out_already_loaded || !out_warmup_more_data) {
        return false;
    }

    *out_target_submodel_id = 0;
    *out_already_loaded = false;
    *out_warmup_more_data = false;

    // 1. Ingest FFT slice and apply log transform
    float X_log[256];
    for (uint32_t i = 0; i < ensemble->raw_bins; i++) {
        X_log[i] = log1pf(fft_raw[i]);
    }

    // 2. Write log-scaled slice to circular buffer (default routed submodel_id is 0 until decided)
    ring_buffer_write(&ensemble->ring_buffer, X_log, 0);

    // 3. Run memory backbone model if present
    if (ensemble->memory_model_loaded) {
        float* mem_input = model_instance_get_input_tensor(&ensemble->memory_model, 0);
        size_t mem_bins = ensemble->memory_model.config.frequency_bins;
        size_t mem_d = ensemble->memory_model.config.d;

        if (mem_bins < ensemble->raw_bins) {
            pool_1d_max_pow2(X_log, ensemble->raw_bins, mem_input, mem_bins);
        } else {
            
            memcpy(mem_input, X_log, mem_bins * sizeof(float));
        }

        memcpy(mem_input + mem_bins, ensemble->h_state, mem_d * sizeof(float));
        memcpy(mem_input + mem_bins + mem_d, ensemble->c_state, mem_d * sizeof(float));

        model_instance_invoke(&ensemble->memory_model);

        const float* mem_output = model_instance_get_output_tensor(&ensemble->memory_model, 0);
        memcpy(ensemble->h_state, mem_output, mem_d * sizeof(float));
        memcpy(ensemble->c_state, mem_output + mem_d, mem_d * sizeof(float));
    }

    // 4. Check Warmup
    ensemble->warmup_steps_done++;
    if (ensemble->warmup_steps_done < ensemble->warmup_steps) {
        *out_warmup_more_data = true;
        return true;
    }

    // 5. Run Router model
    if (!ensemble->router_model_loaded) {
        ESP_LOGE(TAG, "Router model is not loaded");
        return false;
    }

    float* r_input = model_instance_get_input_tensor(&ensemble->router_model, 0);
    uint32_t router_depth = ensemble->router_model.config.temporal_depth;
    uint32_t router_bins = ensemble->router_model.config.frequency_bins;

    // Unroll chronological slices using ring buffer wrapper
    ring_buffer_unroll(&ensemble->ring_buffer, r_input, router_bins, router_depth);

    model_instance_invoke(&ensemble->router_model);

    const float* r_output = model_instance_get_output_tensor(&ensemble->router_model, 0);
    uint32_t num_modes = ensemble->router_model.config.num_modes;
    if (num_modes == 0) {
        return false;
    }

    uint32_t best_mode = 0;
    float max_logit = r_output[0];
    for (uint32_t i = 1; i < num_modes; i++) {
        if (r_output[i] > max_logit) {
            max_logit = r_output[i];
            best_mode = i;
        }
    }

    // Resolve submodel ID from routing table
    uint32_t target_submodel_id = 0;
    for (uint32_t i = 0; i < ensemble->num_routes; i++) {
        if (ensemble->routes[i].out_idx == best_mode) {
            target_submodel_id = ensemble->routes[i].m_id;
            break;
        }
    }

    // Record the resolved submodel ID for the latest written slice
    ring_buffer_set_last_model_id(&ensemble->ring_buffer, target_submodel_id);

    *out_target_submodel_id = target_submodel_id;

    // Query LIFO cache pool. If model is present, gets it (which also moves it to index 0)
    ModelInstance_t* cached = model_cache_get(&ensemble->submodel_cache, target_submodel_id);
    *out_already_loaded = (cached != NULL);

    return true;
}

bool model_ensemble_step_2(model_ensemble_t* ensemble, float* out_anomaly_score, bool* out_is_anomaly) {
    if (!ensemble || !out_anomaly_score || !out_is_anomaly) {
        return false;
    }

    *out_anomaly_score = 0.0f;
    *out_is_anomaly = false;

    // The active submodel is always the most recently used/inserted one, at index 0 of the cache pool
    if (ensemble->submodel_cache.count == 0) {
        ESP_LOGE(TAG, "No submodel loaded in cache pool");
        return false;
    }

    ModelInstance_t* active_submodel = &ensemble->submodel_cache.items[0].model;

    uint32_t s_id = active_submodel->config.model_id;
    uint32_t submodel_depth = active_submodel->config.temporal_depth;
    uint32_t submodel_bins = active_submodel->config.frequency_bins;

    // Verify chronological sequence match via ring buffer wrapper
    if (!ring_buffer_verify_sequence(&ensemble->ring_buffer, s_id, submodel_depth)) {
        ESP_LOGD(TAG, "Submodel %lu sequence broken. Skipping invocation.", (unsigned long)s_id);
        return true;
    }

    float* s_input = model_instance_get_input_tensor(active_submodel, 0);

    // Unroll chronological slices into input tensor
    ring_buffer_unroll(&ensemble->ring_buffer, s_input, submodel_bins, submodel_depth);

    model_instance_invoke(active_submodel);

    // Get the latest slice for target loss calculation
    const float* target_raw = ring_buffer_get_slice(&ensemble->ring_buffer, -1);
    float target_processed[256];

    if (submodel_bins < ensemble->raw_bins) {
        pool_1d_max_pow2(target_raw, ensemble->raw_bins, target_processed, submodel_bins);
    } else {
        memcpy(target_processed, target_raw, submodel_bins * sizeof(float));
    }

    const float* s_output = model_instance_get_output_tensor(active_submodel, 0);
    float loss = compute_reconstruction_loss(target_processed, s_output, submodel_bins, active_submodel->config.loss_mode);
    *out_anomaly_score = loss;
    *out_is_anomaly = (loss >= active_submodel->config.anomaly_threshold);

    return true;
}
