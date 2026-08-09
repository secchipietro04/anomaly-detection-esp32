#include "base_model.h"
#include <stdlib.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_log.h"

static const char* TAG = "BASE_MODEL";

bool base_model_init(base_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config) {
    if (!model || !model_bytes || !config || config->arena_size == 0) {
        return false;
    }
    memset(model, 0, sizeof(base_model_t));
    model->arena_size = config->arena_size;

    /* Handle memory allocation or shared buffer binding */
    if (config->shared_arena != NULL) {
        model->arena_ptr = config->shared_arena;
        model->arena_owned = false;
    } else {
        uint32_t caps = MALLOC_CAP_8BIT | (config->use_psram ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL);
        model->arena_ptr = (uint8_t*)heap_caps_malloc(config->arena_size, caps);
        if (!model->arena_ptr) {
            ESP_LOGE(TAG, "Failed to allocate tensor arena (%zu bytes)", config->arena_size);
            return false;
        }
        model->arena_owned = true;
    }

    return base_model_reload(model, model_bytes);
}

bool base_model_reload(base_model_t* model, const uint8_t* new_model_bytes) {
    if (!model || !new_model_bytes || !model->arena_ptr) {
        return false;
    }

    /* Free existing interpreter instance if loaded */
    if (model->tf_handle != NULL) {
        base_model_unload_interpreter(model);
    }

    /* Bind existing arena to wrapper configuration */
    tflite_config_t tf_cfg = {
        .tensor_arena_size = model->arena_size,
        .tensor_arena = model->arena_ptr,
        .use_psram = false
    };

    tflite_status_t status;
    model->tf_handle = tflite_micro_init(new_model_bytes, &tf_cfg, &status);
    if (!model->tf_handle || status != TFLITE_WRAPPER_OK) {
        ESP_LOGE(TAG, "Failed to load model into interpreter (status: %d)", status);
        return false;
    }

    /* Cache tensor details */
    if (tflite_micro_get_input_details(model->tf_handle, 0, &model->input_details) != TFLITE_WRAPPER_OK ||
        tflite_micro_get_output_details(model->tf_handle, 0, &model->output_details) != TFLITE_WRAPPER_OK) {
        ESP_LOGE(TAG, "Failed to fetch tensor metadata");
        base_model_unload_interpreter(model);
        return false;
    }

    return true;
}

void base_model_unload_interpreter(base_model_t* model) {
    if (!model || !model->tf_handle) return;
    tflite_micro_free(model->tf_handle);
    model->tf_handle = NULL;
}

void base_model_free_arena(base_model_t* model) {
    if (!model) return;
    if (model->arena_owned && model->arena_ptr) {
        heap_caps_free(model->arena_ptr);
    }
    model->arena_ptr = NULL;
    model->arena_owned = false;
}

void base_model_deinit(base_model_t* model) {
    if (!model) return;
    base_model_unload_interpreter(model);
    base_model_free_arena(model);
}

bool base_model_run(base_model_t* model) {
    if (!model || !model->tf_handle) return false;
    return (tflite_micro_invoke(model->tf_handle) == TFLITE_WRAPPER_OK);
}