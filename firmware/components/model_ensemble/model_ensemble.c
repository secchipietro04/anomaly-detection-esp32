#include "model_ensemble.h"
#include <stdlib.h>
#include <string.h>
#include "esp_log.h"

static const char* TAG = "MODEL_ENSEMBLE";

bool model_ensemble_init(model_ensemble_t* ensemble, float confidence_threshold) {
    if (!ensemble) return false;
    memset(ensemble, 0, sizeof(model_ensemble_t));
    ensemble->router_confidence_threshold = confidence_threshold;
    return true;
}

void model_ensemble_deinit(model_ensemble_t* ensemble) {
    if (!ensemble) return;
    memset(ensemble, 0, sizeof(model_ensemble_t));
}

bool model_ensemble_set_router(model_ensemble_t* ensemble, router_model_t* router) {
    if (!ensemble) return false;
    ensemble->router = router;
    ESP_LOGI(TAG, "Router '%s' registered in ensemble", router ? router->name : "NULL");
    return true;
}

bool model_ensemble_run_router(model_ensemble_t* ensemble, const float* input_series, size_t input_len, int* out_best_route_idx, float* out_confidence, fft_shared_data_t* out_shared_fft) {
    if (!ensemble || !input_series || !out_best_route_idx || !out_confidence) return false;

    *out_best_route_idx = -1;
    *out_confidence = 0.0f;

    if (!ensemble->router) {
        ESP_LOGE(TAG, "No router registered in the ensemble");
        return false;
    }

    size_t r_input_len = ensemble->router->input_length;
    if (r_input_len > input_len) {
        r_input_len = input_len;
    }

    // Run router and populate the caller's shared FFT target buffer (if provided)
    if (!ensemble->router->run(ensemble->router, input_series, r_input_len, out_best_route_idx, out_confidence, out_shared_fft)) {
        ESP_LOGE(TAG, "Router execution failed");
        return false;
    }

    return true;
}

bool model_ensemble_load_submodel(model_ensemble_t* ensemble, anomaly_model_t* submodel, int route_idx, float anomaly_threshold) {
    if (!ensemble || !submodel || route_idx < 0 || route_idx >= MAX_SUBMODELS) return false;

    // Store the anomaly threshold directly inside the polymorphic submodel structure
    submodel->anomaly_threshold = anomaly_threshold;

    // Load/swap the submodel in the slot corresponding to the router class
    ensemble->submodels[route_idx] = submodel;
    ESP_LOGI(TAG, "Submodel '%s' loaded into slot %d (threshold: %.4f)", 
             submodel->name, route_idx, anomaly_threshold);
    return true;
}

bool model_ensemble_unload_submodel(model_ensemble_t* ensemble, uint32_t model_id) {
    if (!ensemble) return false;

    bool found = false;
    for (int i = 0; i < MAX_SUBMODELS; i++) {
        if (ensemble->submodels[i] && ensemble->submodels[i]->model_id == model_id) {
            ESP_LOGI(TAG, "Unloading submodel '%s' from slot %d", ensemble->submodels[i]->name, i);
            ensemble->submodels[i]->anomaly_threshold = 0.0f;
            ensemble->submodels[i] = NULL;
            found = true;
        }
    }

    if (!found) {
        ESP_LOGW(TAG, "Submodel ID %lu was not loaded in the ensemble", model_id);
    }
    return found;
}

bool model_ensemble_run_submodel(model_ensemble_t* ensemble, uint32_t model_id, const float* input_series, size_t input_len, const fft_shared_data_t* shared_fft, float* out_anomaly_score, bool* out_is_anomaly) {
    if (!ensemble || !input_series || !out_anomaly_score || !out_is_anomaly) return false;

    *out_anomaly_score = 0.0f;
    *out_is_anomaly = false;

    // Find the slot containing the submodel with the requested ID
    int slot = -1;
    for (int i = 0; i < MAX_SUBMODELS; i++) {
        if (ensemble->submodels[i] && ensemble->submodels[i]->model_id == model_id) {
            slot = i;
            break;
        }
    }

    if (slot == -1) {
        ESP_LOGE(TAG, "Execution failed: Model ID %lu is not currently loaded in the ensemble!", model_id);
        return false;
    }

    anomaly_model_t* active_model = ensemble->submodels[slot];

    // Run the active submodel. We forward the caller's precalculated shared_fft.
    // If it is NULL or marked invalid, the submodel will compute its own FFT.
    if (!active_model->run(active_model, input_series, input_len, shared_fft, out_anomaly_score)) {
        ESP_LOGE(TAG, "Submodel '%s' execution failed", active_model->name);
        return false;
    }

    // Set the anomaly flag relative to the model's own threshold
    *out_is_anomaly = (*out_anomaly_score >= active_model->anomaly_threshold);
    return true;
}
