#include "base_model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "ops.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include <new>

static const char* TAG = "BASE_MODEL";

// Context wrapping the resolver and interpreter next to each other
// to allow cleanup via container_of pattern in C deinit.
struct ModelContext {
    tflite::MicroMutableOpResolver<kOpResolverSize> resolver;
    tflite::MicroInterpreter interpreter;

    ModelContext(const tflite::Model* model, uint8_t* arena, size_t arena_size)
        : interpreter(model, resolver, arena, arena_size) {
        register_configured_ops(resolver);
    }
};

extern "C" int model_instance_init(ModelInstance_t* self, const uint8_t* model_data, size_t model_size, size_t arena_size) {
    if (!self || !model_data || model_size == 0 || arena_size == 0) {
        return MODEL_ERR_GENERIC;
    }

    // Allocate a dedicated region of memory to copy the model flatbuffer into
    uint8_t* model_copy = (uint8_t*)malloc(model_size);
    if (!model_copy) {
        ESP_LOGE(TAG, "Failed to allocate dedicated model binary copy buffer (%zu bytes)", model_size);
        return MODEL_ERR_NO_MEM;
    }
    memcpy(model_copy, model_data, model_size);
    self->model_data = model_copy;

    const tflite::Model* model = tflite::GetModel(model_copy);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model version schema mismatch!");
        free(model_copy);
        self->model_data = nullptr;
        return MODEL_ERR_SCHEMA;
    }

    self->arena_size = arena_size;
    if (!self->tensor_arena) {
        // Try internal RAM allocation first for speed, fallback to PSRAM if size demands
        self->tensor_arena = (uint8_t*)heap_caps_malloc(arena_size, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!self->tensor_arena) {
            self->tensor_arena = (uint8_t*)heap_caps_malloc(arena_size, MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
        }
        if (!self->tensor_arena) {
            ESP_LOGE(TAG, "Failed to allocate tensor arena of size %zu", arena_size);
            free(model_copy);
            self->model_data = nullptr;
            return MODEL_ERR_NO_MEM;
        }
    }

    // Allocate adjacent memory block for the mutable resolver and the interpreter
    ModelContext* ctx = new (std::nothrow) ModelContext(model, self->tensor_arena, arena_size);
    if (!ctx) {
        ESP_LOGE(TAG, "Failed to allocate TFLM model execution context");
        if (self->tensor_arena) {
            heap_caps_free(self->tensor_arena);
            self->tensor_arena = nullptr;
        }
        free(model_copy);
        self->model_data = nullptr;
        return MODEL_ERR_NO_MEM;
    }

    TfLiteStatus alloc_status = ctx->interpreter.AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        ESP_LOGE(TAG, "TFLM AllocateTensors() failed with code %d", alloc_status);
        delete ctx;
        if (self->tensor_arena) {
            heap_caps_free(self->tensor_arena);
            self->tensor_arena = nullptr;
        }
        free(model_copy);
        self->model_data = nullptr;
        return MODEL_ERR_TFLM;
    }

    self->interpreter = &ctx->interpreter;
    
    // Bind the struct interface pointers
    self->init = model_instance_init;
    self->deinit = model_instance_deinit;
    self->invoke = model_instance_invoke;

    return MODEL_SUCCESS;
}

extern "C" void model_instance_deinit(ModelInstance_t* self) {
    if (!self || !self->interpreter) {
        return;
    }

    // Find starting pointer of ModelContext from the embedded interpreter field offset
    char* interpreter_ptr = (char*)self->interpreter;
    size_t offset = offsetof(ModelContext, interpreter);
    ModelContext* ctx = (ModelContext*)(interpreter_ptr - offset);

    delete ctx;
    self->interpreter = nullptr;

    if (self->tensor_arena) {
        heap_caps_free(self->tensor_arena);
        self->tensor_arena = nullptr;
    }

    if (self->model_data) {
        free((void*)self->model_data);
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
    TfLiteTensor* tensor = self->interpreter->input(index);
    if (!tensor) return nullptr;
    return tensor->data.f;
}

extern "C" const float* model_instance_get_output_tensor(ModelInstance_t* self, size_t index) {
    if (!self || !self->interpreter) return nullptr;
    TfLiteTensor* tensor = self->interpreter->output(index);
    if (!tensor) return nullptr;
    return tensor->data.f;
}

extern "C" size_t model_instance_get_input_size(ModelInstance_t* self, size_t index) {
    if (!self || !self->interpreter) return 0;
    TfLiteTensor* tensor = self->interpreter->input(index);
    if (!tensor) return 0;
    return tensor->bytes / sizeof(float);
}

extern "C" size_t model_instance_get_output_size(ModelInstance_t* self, size_t index) {
    if (!self || !self->interpreter) return 0;
    TfLiteTensor* tensor = self->interpreter->output(index);
    if (!tensor) return 0;
    return tensor->bytes / sizeof(float);
}
