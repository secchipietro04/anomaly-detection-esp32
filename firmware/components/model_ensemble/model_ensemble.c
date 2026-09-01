#include "model_ensemble.h"
#include "dsp_utils.h"
#include "loss_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"

static const char* TAG = "MODEL_ENSEMBLE";

void model_ensemble_lock(model_ensemble_t* ensemble) {
    if (ensemble && ensemble->mutex) {
        xSemaphoreTakeRecursive(ensemble->mutex, portMAX_DELAY);
    }
}

void model_ensemble_unlock(model_ensemble_t* ensemble) {
    if (ensemble && ensemble->mutex) {
        xSemaphoreGiveRecursive(ensemble->mutex);
    }
}

bool model_ensemble_init(model_ensemble_t* ensemble, uint32_t raw_bins, uint32_t history_depth, uint32_t warmup_steps) {
    if (!ensemble) return false;
    memset(ensemble, 0, sizeof(model_ensemble_t));

    ensemble->mutex = xSemaphoreCreateRecursiveMutex();

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
    model_ensemble_lock(ensemble);

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
    
    model_ensemble_unlock(ensemble);
    if (ensemble->mutex) {
        vSemaphoreDelete(ensemble->mutex);
        ensemble->mutex = NULL;
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

    model_ensemble_lock(ensemble);

    if (ensemble->router_model_loaded) {
        model_instance_deinit(&ensemble->router_model);
        ensemble->router_model_loaded = false;
    }

    int ret = model_instance_init(&ensemble->router_model, model_data, model_size, ensemble->router_arena_size);
    if (ret != MODEL_SUCCESS) {
        ESP_LOGE(TAG, "Failed to load router model instance");
        model_ensemble_unlock(ensemble);
        return false;
    }

    ensemble->router_model_loaded = true;
    model_ensemble_unlock(ensemble);
    return true;
}

bool model_ensemble_load_memory(model_ensemble_t* ensemble, const uint8_t* model_data, size_t model_size) {
    if (!ensemble || !model_data || model_size == 0) return false;

    model_ensemble_lock(ensemble);

    if (ensemble->memory_model_loaded) {
        model_instance_deinit(&ensemble->memory_model);
        ensemble->memory_model_loaded = false;
    }

    int ret = model_instance_init(&ensemble->memory_model, model_data, model_size, ensemble->memory_arena_size);
    if (ret != MODEL_SUCCESS) {
        ESP_LOGE(TAG, "Failed to load memory backbone model instance");
        model_ensemble_unlock(ensemble);
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
        model_ensemble_unlock(ensemble);
        return false;
    }

    model_ensemble_unlock(ensemble);
    return true;
}

void model_ensemble_set_routes(model_ensemble_t* ensemble, const route_entry_t* routes, uint32_t num_routes) {
    if (!ensemble || !routes) return;

    model_ensemble_lock(ensemble);
    ensemble->num_routes = num_routes < MAX_ROUTES ? num_routes : MAX_ROUTES;
    for (uint32_t i = 0; i < ensemble->num_routes; i++) {
        ensemble->routes[i] = routes[i];
    }
    model_ensemble_unlock(ensemble);
}

int model_ensemble_load_submodel(model_ensemble_t* ensemble, const uint8_t* model_data, size_t model_size) {
    if (!ensemble || !model_data || model_size == 0) return MODEL_ERR_GENERIC;

    model_ensemble_lock(ensemble);
    int ret = model_cache_insert(&ensemble->submodel_cache, model_data, model_size);
    model_ensemble_unlock(ensemble);
    return ret;
}

void model_ensemble_set_cache_capacity(model_ensemble_t* ensemble, uint32_t capacity) {
    if (!ensemble) return;
    model_ensemble_lock(ensemble);
    model_cache_set_capacity(&ensemble->submodel_cache, capacity);
    model_ensemble_unlock(ensemble);
}

bool model_ensemble_inf_memory(model_ensemble_t* ensemble, const float* log_features) {
    if (!ensemble || !log_features) {
        return false;
    }

    model_ensemble_lock(ensemble);
    if (ensemble->memory_model_loaded) {
        float* mem_input = model_instance_get_input_tensor(&ensemble->memory_model, 0);
        size_t mem_bins = ensemble->memory_model.config.frequency_bins;
        size_t mem_d = ensemble->memory_model.config.d;

        if (mem_bins < ensemble->raw_bins) {
            pool_1d_max_pow2(log_features, ensemble->raw_bins, mem_input, mem_bins);
        } else {
            memcpy(mem_input, log_features, mem_bins * sizeof(float));
        }

        memcpy(mem_input + mem_bins, ensemble->h_state, mem_d * sizeof(float));
        memcpy(mem_input + mem_bins + mem_d, ensemble->c_state, mem_d * sizeof(float));

        model_instance_invoke(&ensemble->memory_model);

        const float* mem_output = model_instance_get_output_tensor(&ensemble->memory_model, 0);
        memcpy(ensemble->h_state, mem_output, mem_d * sizeof(float));
        memcpy(ensemble->c_state, mem_output + mem_d, mem_d * sizeof(float));
    }
    model_ensemble_unlock(ensemble);

    return true;
}

bool model_ensemble_inf_router(model_ensemble_t* ensemble, const float* spec, uint32_t num_frames, uint32_t* out_target_submodel_id, bool* out_already_loaded, bool* out_is_anomaly) {
    if (!ensemble || !spec || !out_target_submodel_id || !out_already_loaded || !out_is_anomaly) {
        return false;
    }

    *out_target_submodel_id = 0;
    *out_already_loaded = false;
    *out_is_anomaly = false;

    model_ensemble_lock(ensemble);

    if (!ensemble->router_model_loaded) {
        ESP_LOGE(TAG, "Router model is not loaded");
        model_ensemble_unlock(ensemble);
        return false;
    }

    float* r_input = model_instance_get_input_tensor(&ensemble->router_model, 0);
    if (!r_input) {
        ESP_LOGE(TAG, "Router input tensor is NULL");
        model_ensemble_unlock(ensemble);
        return false;
    }

    uint32_t router_depth = ensemble->router_model.config.temporal_depth > 0 ? ensemble->router_model.config.temporal_depth : 8;
    uint32_t router_bins = ensemble->router_model.config.frequency_bins > 0 ? ensemble->router_model.config.frequency_bins : ensemble->raw_bins;
    if (router_depth > num_frames) router_depth = num_frames;

    size_t max_router_floats = model_instance_get_input_size(&ensemble->router_model, 0);

    // Unroll chronological slices directly from linear spectrogram buffer
    for (uint32_t k = 0; k < router_depth; k++) {
        size_t offset = k * router_bins;
        if (max_router_floats > 0 && (offset + router_bins > max_router_floats)) break;

        const float* src = spec + (k * ensemble->raw_bins);
        float* dest = r_input + offset;
        if (router_bins < ensemble->raw_bins) {
            pool_1d_max_pow2(src, ensemble->raw_bins, dest, router_bins);
        } else {
            memcpy(dest, src, router_bins * sizeof(float));
        }
    }

    model_instance_invoke(&ensemble->router_model);

    const float* r_output = model_instance_get_output_tensor(&ensemble->router_model, 0);
    if (!r_output) {
        ESP_LOGE(TAG, "Router output tensor is NULL");
        model_ensemble_unlock(ensemble);
        return false;
    }

    uint32_t num_modes = ensemble->router_model.config.num_modes;
    if (num_modes == 0) {
        model_ensemble_unlock(ensemble);
        return false;
    }

    // Find best mode and calculate Softmax probability
    uint32_t best_mode = 0;
    float max_logit = r_output[0];
    for (uint32_t i = 1; i < num_modes; i++) {
        if (r_output[i] > max_logit) {
            max_logit = r_output[i];
            best_mode = i;
        }
    }

    float sum_exp = 0.0f;
    for (uint32_t i = 0; i < num_modes; i++) {
        sum_exp += expf(r_output[i]);
    }
    float best_prob = expf(max_logit) / sum_exp;

    // If max probability is below the anomaly threshold, classify as router anomaly
    if (best_prob < ROUTER_CONFIDENCE_THRESHOLD) {
        ESP_LOGW(TAG, "Router confidence %f below threshold %f. Segment is anomalous.", best_prob, ROUTER_CONFIDENCE_THRESHOLD);
        *out_is_anomaly = true;
        model_ensemble_unlock(ensemble);
        return true;
    }

    // Resolve submodel ID from routing table
    uint32_t target_submodel_id = 0;
    bool found_route = false;
    for (uint32_t i = 0; i < ensemble->num_routes; i++) {
        if (ensemble->routes[i].out_idx == best_mode) {
            target_submodel_id = ensemble->routes[i].m_id;
            found_route = true;
            break;
        }
    }

    if (!found_route || target_submodel_id == 0) {
        ESP_LOGW(TAG, "Router best mode %u has no route mapping. Classified as Router Anomaly!", (unsigned int)best_mode);
        *out_is_anomaly = true;
        *out_target_submodel_id = 0;
        model_ensemble_unlock(ensemble);
        return true;
    }

    *out_target_submodel_id = target_submodel_id;

    // Query LIFO cache pool. If model is present, gets it (which also moves it to index 0)
    ModelInstance_t* cached = model_cache_get(&ensemble->submodel_cache, target_submodel_id);
    *out_already_loaded = (cached != NULL);

    model_ensemble_unlock(ensemble);
    return true;
}

bool model_ensemble_inf_ae(model_ensemble_t* ensemble, const float* spec, uint32_t num_frames, float* out_anomaly_score, bool* out_is_anomaly, uint32_t skip_amount) {
    if (!ensemble || !spec || !out_anomaly_score || !out_is_anomaly) {
        return false;
    }

    *out_anomaly_score = 0.0f;
    *out_is_anomaly = false;

    model_ensemble_lock(ensemble);

    // The active submodel is always the most recently used/inserted one, at index 0 of the cache pool
    if (ensemble->submodel_cache.count == 0) {
        ESP_LOGE(TAG, "No submodel loaded in cache pool");
        model_ensemble_unlock(ensemble);
        return false;
    }

    ModelInstance_t* active_submodel = &ensemble->submodel_cache.items[0].model;
    if (!active_submodel->interpreter) {
        ESP_LOGE(TAG, "Active submodel interpreter is NULL");
        model_ensemble_unlock(ensemble);
        return false;
    }

    uint32_t depth = active_submodel->config.temporal_depth > 0 ? active_submodel->config.temporal_depth : 8;
    uint32_t bins = active_submodel->config.frequency_bins > 0 ? active_submodel->config.frequency_bins : ensemble->raw_bins;

    float* s_input = model_instance_get_input_tensor(active_submodel, 0);
    if (!s_input) {
        ESP_LOGE(TAG, "Submodel input tensor is NULL");
        model_ensemble_unlock(ensemble);
        return false;
    }

    size_t max_input_floats = model_instance_get_input_size(active_submodel, 0);
    if (max_input_floats > 0 && bins > 0) {
        uint32_t model_depth = (uint32_t)(max_input_floats / bins);
        if (model_depth > 0 && model_depth < depth) {
            depth = model_depth;
        }
    }

    // Determine evaluation loop range based on whether it is a memory/recurrent model
    int start_t;
    int end_t = (int)num_frames - (int)depth;
    if (end_t < 0) {
        end_t = 0;
        depth = num_frames;
    }
    bool is_memory = ensemble->memory_model_loaded;

    if (is_memory) {
        start_t = end_t;
    } else {
        start_t = 0;
    }

    float total_loss = 0.0f;
    int evaluations = 0;
    bool any_anomaly = false;

    int t = start_t;
    while (t <= end_t) {
        // Copy/unroll the window of frames starting at frame t
        for (uint32_t k = 0; k < depth; k++) {
            size_t offset = k * bins;
            if (max_input_floats > 0 && (offset + bins > max_input_floats)) break;

            const float* src_slice = spec + ((t + k) * ensemble->raw_bins);
            float* dest_ptr = s_input + offset;
            if (bins < ensemble->raw_bins) {
                pool_1d_max_pow2(src_slice, ensemble->raw_bins, dest_ptr, bins);
            } else {
                memcpy(dest_ptr, src_slice, bins * sizeof(float));
            }
        }

        model_instance_invoke(active_submodel);

        // Get the target slice for reconstruction loss calculation
        const float* target_raw = spec + ((t + depth - 1) * ensemble->raw_bins);
        float target_processed[256];
        size_t safe_bins = bins > 256 ? 256 : bins;
        if (bins < ensemble->raw_bins) {
            pool_1d_max_pow2(target_raw, ensemble->raw_bins, target_processed, bins);
        } else {
            memcpy(target_processed, target_raw, safe_bins * sizeof(float));
        }

        const float* s_output = model_instance_get_output_tensor(active_submodel, 0);
        if (!s_output) {
            t += (int)depth + (int)skip_amount;
            continue;
        }

        float loss = compute_reconstruction_loss(target_processed, s_output, bins, active_submodel->config.loss_mode);

        total_loss += loss;
        if (loss >= active_submodel->config.anomaly_threshold) {
            any_anomaly = true;
        }
        evaluations++;

        if (is_memory) {
            break;
        }
        t += (int)depth + (int)skip_amount;
    }

    if (evaluations > 0) {
        *out_anomaly_score = total_loss / evaluations;
        *out_is_anomaly = any_anomaly;
    }

    model_ensemble_unlock(ensemble);
    return true;
}
