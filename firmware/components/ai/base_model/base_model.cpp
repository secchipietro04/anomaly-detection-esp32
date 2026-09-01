#include "base_model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "ops.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include <new>

static const char* TAG = "BASE_MODEL";

#include <memory>

// Context wrapping the resolver and interpreter next to each other
struct ModelContext {
    tflite::MicroMutableOpResolver<kOpResolverSize> resolver;
    std::unique_ptr<tflite::MicroInterpreter> interpreter;

    ModelContext(const tflite::Model* model, uint8_t* arena, size_t arena_size) {
        register_configured_ops(resolver);
        interpreter = std::unique_ptr<tflite::MicroInterpreter>(new tflite::MicroInterpreter(model, resolver, arena, arena_size));
    }
};

extern "C" int model_instance_init(ModelInstance_t* self, const uint8_t* model_data, size_t model_size, size_t arena_size) {
    if (!self || !model_data || model_size == 0 || arena_size == 0) {
        return MODEL_ERR_GENERIC;
    }

    // Allocate in PSRAM first, fallback to internal RAM with manual 16-byte alignment
    void* model_raw = heap_caps_malloc(model_size + 16, MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    if (!model_raw) {
        model_raw = heap_caps_malloc(model_size + 16, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    }
    if (!model_raw) {
        ESP_LOGE(TAG, "Failed to allocate dedicated model binary copy buffer (%zu bytes)", model_size);
        return MODEL_ERR_NO_MEM;
    }

    uint8_t* model_copy = (uint8_t*)(((uintptr_t)model_raw + 15) & ~((uintptr_t)15));
    memcpy(model_copy, model_data, model_size);
    self->model_data_raw = model_raw;
    self->model_data = model_copy;

    const tflite::Model* model = tflite::GetModel(model_copy);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model version schema mismatch!");
        heap_caps_free(model_raw);
        self->model_data_raw = nullptr;
        self->model_data = nullptr;
        return MODEL_ERR_SCHEMA;
    }

    // Adaptive arena allocation retry loop: try increasing arena size up to 512KB
    size_t current_arena_size = arena_size;
    const size_t max_arena_size = 512 * 1024;
    ModelContext* ctx = nullptr;

    while (current_arena_size <= max_arena_size) {
        if (self->arena_raw) {
            heap_caps_free(self->arena_raw);
            self->arena_raw = nullptr;
            self->tensor_arena = nullptr;
        }

        // Allocate 4096 bytes extra as a safety buffer so TFLM tail allocator never touches SRAM boundaries
        size_t alloc_bytes = current_arena_size + 4096;
        void* arena_raw = heap_caps_malloc(alloc_bytes, MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
        if (!arena_raw) {
            arena_raw = heap_caps_malloc(alloc_bytes, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        }

        if (!arena_raw) {
            ESP_LOGW(TAG, "Heap allocation failed for arena size %zu", current_arena_size);
            break;
        }

        uint8_t* tensor_arena = (uint8_t*)(((uintptr_t)arena_raw + 15) & ~((uintptr_t)15));
        self->arena_raw = arena_raw;
        self->tensor_arena = tensor_arena;

        if (ctx) {
            delete ctx;
            ctx = nullptr;
        }

        ctx = new (std::nothrow) ModelContext(model, self->tensor_arena, current_arena_size);
        if (!ctx) {
            ESP_LOGE(TAG, "Failed to allocate TFLM context for arena %zu", current_arena_size);
            break;
        }

        TfLiteStatus alloc_status = ctx->interpreter->AllocateTensors();
        if (alloc_status == kTfLiteOk) {
            self->arena_size = current_arena_size;
            self->context = ctx;
            self->interpreter = ctx->interpreter.get();
            self->init = model_instance_init;
            self->deinit = model_instance_deinit;
            self->invoke = model_instance_invoke;
            ESP_LOGI(TAG, "TFLM model initialized successfully with arena size %zu bytes (used: %zu bytes)",
                     current_arena_size, ctx->interpreter->arena_used_bytes());
            return MODEL_SUCCESS;
        }

        ESP_LOGW(TAG, "AllocateTensors() failed with arena %zu bytes (status %d). Retrying with larger arena...",
                 current_arena_size, alloc_status);
        current_arena_size *= 2;
    }

    if (ctx) {
        delete ctx;
    }
    if (self->arena_raw) {
        heap_caps_free(self->arena_raw);
        self->arena_raw = nullptr;
        self->tensor_arena = nullptr;
    }
    if (self->model_data_raw) {
        heap_caps_free(self->model_data_raw);
        self->model_data_raw = nullptr;
        self->model_data = nullptr;
    }
    return MODEL_ERR_TFLM;
}

extern "C" void model_instance_deinit(ModelInstance_t* self) {
    if (!self) {
        return;
    }

    if (self->context) {
        ModelContext* ctx = static_cast<ModelContext*>(self->context);
        delete ctx;
        self->context = nullptr;
    }
    self->interpreter = nullptr;

    if (self->arena_raw) {
        heap_caps_free(self->arena_raw);
        self->arena_raw = nullptr;
        self->tensor_arena = nullptr;
    }

    if (self->model_data_raw) {
        heap_caps_free(self->model_data_raw);
        self->model_data_raw = nullptr;
        self->model_data = nullptr;
    }
}

extern "C" void model_instance_invoke(ModelInstance_t* self) {
    if (!self || !self->interpreter) {
        return;
    }

    TfLiteStatus status = self->interpreter->Invoke();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "TFLite interpreter execution failed with status %d", status);
    }
}

extern "C" float* model_instance_get_input_tensor(ModelInstance_t* self, size_t index) {
    if (!self || !self->interpreter) return nullptr;
    if (self->interpreter->inputs_size() <= index) return nullptr;
    TfLiteTensor* tensor = self->interpreter->input(index);
    if (!tensor) return nullptr;
    return tensor->data.f;
}

extern "C" size_t model_instance_get_input_size(ModelInstance_t* self, size_t index) {
    if (!self || !self->interpreter) return 0;
    if (self->interpreter->inputs_size() <= index) return 0;
    TfLiteTensor* tensor = self->interpreter->input(index);
    if (!tensor) return 0;
    return tensor->bytes / sizeof(float);
}

extern "C" const float* model_instance_get_output_tensor(ModelInstance_t* self, size_t index) {
    if (!self || !self->interpreter) return nullptr;
    if (self->interpreter->outputs_size() <= index) return nullptr;
    TfLiteTensor* tensor = self->interpreter->output(index);
    if (!tensor) return nullptr;
    return tensor->data.f;
}

extern "C" size_t model_instance_get_output_size(ModelInstance_t* self, size_t index) {
    if (!self || !self->interpreter) return 0;
    if (self->interpreter->outputs_size() <= index) return 0;
    TfLiteTensor* tensor = self->interpreter->output(index);
    if (!tensor) return 0;
    return tensor->bytes / sizeof(float);
}
