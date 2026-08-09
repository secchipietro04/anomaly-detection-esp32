#include "tf_wrapper.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "ops.hpp"

#include "esp_heap_caps.h"
#include "esp_log.h"

static const char* TAG = "TFLITE_WRAPPER";

struct tflite_context_t {
    const tflite::Model* model;
    tflite::MicroMutableOpResolver<kOpResolverSize>* resolver;
    tflite::MicroInterpreter* interpreter;
    uint8_t* tensor_arena;
    bool arena_allocated_internally;
};

extern "C" {

tflite_handle_t tflite_micro_init(const uint8_t* model_buffer, const tflite_config_t* config, tflite_status_t* status) {
    if (!model_buffer || !config || config->tensor_arena_size == 0) {
        if (status) *status = TFLITE_WRAPPER_ERROR_INVALID_PARAM;
        return NULL;
    }

    tflite_context_t* ctx = (tflite_context_t*)calloc(1, sizeof(tflite_context_t));
    if (!ctx) {
        if (status) *status = TFLITE_WRAPPER_ERROR_ALLOCATION;
        return NULL;
    }

    ctx->model = tflite::GetModel(model_buffer);
    if (ctx->model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch!");
        free(ctx);
        if (status) *status = TFLITE_WRAPPER_ERROR_INIT_MODEL;
        return NULL;
    }

    if (config->tensor_arena != NULL) {
        ctx->tensor_arena = config->tensor_arena;
        ctx->arena_allocated_internally = false;
    } else {
        uint32_t caps = MALLOC_CAP_8BIT;
        caps |= config->use_psram ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL;
        ctx->tensor_arena = (uint8_t*)heap_caps_malloc(config->tensor_arena_size, caps);
        if (!ctx->tensor_arena) {
            ESP_LOGE(TAG, "Failed to allocate tensor arena (%zu bytes)", config->tensor_arena_size);
            free(ctx);
            if (status) *status = TFLITE_WRAPPER_ERROR_ALLOCATION;
            return NULL;
        }
        ctx->arena_allocated_internally = true;
    }

    ctx->resolver = new tflite::MicroMutableOpResolver<kOpResolverSize>();
    register_configured_ops(*ctx->resolver);

    ctx->interpreter = new tflite::MicroInterpreter(
        ctx->model,
        *ctx->resolver,
        ctx->tensor_arena,
        config->tensor_arena_size
    );

    TfLiteStatus alloc_status = ctx->interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed with status code %d", alloc_status);
        tflite_micro_free(ctx);
        if (status) *status = TFLITE_WRAPPER_ERROR_INTERPRETER;
        return NULL;
    }

    if (status) *status = TFLITE_WRAPPER_OK;
    return (tflite_handle_t)ctx;
}

tflite_status_t tflite_micro_invoke(tflite_handle_t handle) {
    if (!handle) return TFLITE_WRAPPER_ERROR_INVALID_PARAM;
    tflite_context_t* ctx = (tflite_context_t*)handle;
    return (ctx->interpreter->Invoke() == kTfLiteOk) ? TFLITE_WRAPPER_OK : TFLITE_WRAPPER_ERROR_INVOKE;
}

tflite_status_t tflite_micro_get_input_details(tflite_handle_t handle, size_t index, tflite_tensor_details_t* details) {
    if (!handle || !details) return TFLITE_WRAPPER_ERROR_INVALID_PARAM;
    tflite_context_t* ctx = (tflite_context_t*)handle;

    TfLiteTensor* tensor = ctx->interpreter->input(index);
    if (!tensor) return TFLITE_WRAPPER_ERROR_INVALID_PARAM;

    details->data = tensor->data.raw;
    details->bytes = tensor->bytes;
    details->type = tensor->type;
    details->scale = tensor->params.scale;
    details->zero_point = tensor->params.zero_point;
    details->num_dims = tensor->dims->size;
    for (size_t i = 0; i < details->num_dims && i < 4; i++) {
        details->dims[i] = tensor->dims->data[i];
    }
    return TFLITE_WRAPPER_OK;
}

tflite_status_t tflite_micro_get_output_details(tflite_handle_t handle, size_t index, tflite_tensor_details_t* details) {
    if (!handle || !details) return TFLITE_WRAPPER_ERROR_INVALID_PARAM;
    tflite_context_t* ctx = (tflite_context_t*)handle;

    TfLiteTensor* tensor = ctx->interpreter->output(index);
    if (!tensor) return TFLITE_WRAPPER_ERROR_INVALID_PARAM;

    details->data = tensor->data.raw;
    details->bytes = tensor->bytes;
    details->type = tensor->type;
    details->scale = tensor->params.scale;
    details->zero_point = tensor->params.zero_point;
    details->num_dims = tensor->dims->size;
    for (size_t i = 0; i < details->num_dims && i < 4; i++) {
        details->dims[i] = tensor->dims->data[i];
    }
    return TFLITE_WRAPPER_OK;
}

void tflite_micro_free(tflite_handle_t handle) {
    if (!handle) return;
    tflite_context_t* ctx = (tflite_context_t*)handle;

    delete ctx->interpreter;
    delete ctx->resolver;
    if (ctx->arena_allocated_internally && ctx->tensor_arena) {
        heap_caps_free(ctx->tensor_arena);
    }
    free(ctx);
}

static const char* enabled_ops[] = {
#ifdef CONFIG_TFL_USE_ABS
    "ABS",
#endif
#ifdef CONFIG_TFL_USE_ADD
    "ADD",
#endif
#ifdef CONFIG_TFL_USE_ADD_N
    "ADD_N",
#endif
#ifdef CONFIG_TFL_USE_ARG_MAX
    "ARG_MAX",
#endif
#ifdef CONFIG_TFL_USE_ARG_MIN
    "ARG_MIN",
#endif
#ifdef CONFIG_TFL_USE_ASSIGN_VARIABLE
    "ASSIGN_VARIABLE",
#endif
#ifdef CONFIG_TFL_USE_AVERAGE_POOL_2D
    "AVERAGE_POOL_2D",
#endif
#ifdef CONFIG_TFL_USE_BATCH_MATMUL
    "BATCH_MATMUL",
#endif
#ifdef CONFIG_TFL_USE_BATCH_TO_SPACE_ND
    "BATCH_TO_SPACE_ND",
#endif
#ifdef CONFIG_TFL_USE_BROADCAST_ARGS
    "BROADCAST_ARGS",
#endif
#ifdef CONFIG_TFL_USE_BROADCAST_TO
    "BROADCAST_TO",
#endif
#ifdef CONFIG_TFL_USE_CALL_ONCE
    "CALL_ONCE",
#endif
#ifdef CONFIG_TFL_USE_CAST
    "CAST",
#endif
#ifdef CONFIG_TFL_USE_CEIL
    "CEIL",
#endif
#ifdef CONFIG_TFL_USE_CIRCULAR_BUFFER
    "CIRCULAR_BUFFER",
#endif
#ifdef CONFIG_TFL_USE_CONCATENATION
    "CONCATENATION",
#endif
#ifdef CONFIG_TFL_USE_CONV_2D
    "CONV_2D",
#endif
#ifdef CONFIG_TFL_USE_COS
    "COS",
#endif
#ifdef CONFIG_TFL_USE_CUMSUM
    "CUMSUM",
#endif
#ifdef CONFIG_TFL_USE_DECODE
    "DECODE",
#endif
#ifdef CONFIG_TFL_USE_DELAY
    "DELAY",
#endif
#ifdef CONFIG_TFL_USE_DEPTH_TO_SPACE
    "DEPTH_TO_SPACE",
#endif
#ifdef CONFIG_TFL_USE_DEPTHWISE_CONV_2D
    "DEPTHWISE_CONV_2D",
#endif
#ifdef CONFIG_TFL_USE_DEQUANTIZE
    "DEQUANTIZE",
#endif
#ifdef CONFIG_TFL_USE_DETECTION_POSTPROCESS
    "DETECTION_POSTPROCESS",
#endif
#ifdef CONFIG_TFL_USE_DIV
    "DIV",
#endif
#ifdef CONFIG_TFL_USE_DYNAMIC_UPDATE_SLICE
    "DYNAMIC_UPDATE_SLICE",
#endif
#ifdef CONFIG_TFL_USE_EMBEDDING_LOOKUP
    "EMBEDDING_LOOKUP",
#endif
#ifdef CONFIG_TFL_USE_ENERGY
    "ENERGY",
#endif
#ifdef CONFIG_TFL_USE_ELU
    "ELU",
#endif
#ifdef CONFIG_TFL_USE_EQUAL
    "EQUAL",
#endif
#ifdef CONFIG_TFL_USE_ETHOSU
    "ETHOSU",
#endif
#ifdef CONFIG_TFL_USE_EXP
    "EXP",
#endif
#ifdef CONFIG_TFL_USE_EXPAND_DIMS
    "EXPAND_DIMS",
#endif
#ifdef CONFIG_TFL_USE_FFT_AUTO_SCALE
    "FFT_AUTO_SCALE",
#endif
#ifdef CONFIG_TFL_USE_FILL
    "FILL",
#endif
#ifdef CONFIG_TFL_USE_FILTER_BANK
    "FILTER_BANK",
#endif
#ifdef CONFIG_TFL_USE_FILTER_BANK_LOG
    "FILTER_BANK_LOG",
#endif
#ifdef CONFIG_TFL_USE_FILTER_BANK_SQUARE_ROOT
    "FILTER_BANK_SQUARE_ROOT",
#endif
#ifdef CONFIG_TFL_USE_FILTER_BANK_SPECTRAL_SUBTRACTION
    "FILTER_BANK_SPECTRAL_SUBTRACTION",
#endif
#ifdef CONFIG_TFL_USE_FLOOR
    "FLOOR",
#endif
#ifdef CONFIG_TFL_USE_FLOOR_DIV
    "FLOOR_DIV",
#endif
#ifdef CONFIG_TFL_USE_FLOOR_MOD
    "FLOOR_MOD",
#endif
#ifdef CONFIG_TFL_USE_FRAMER
    "FRAMER",
#endif
#ifdef CONFIG_TFL_USE_FULLY_CONNECTED
    "FULLY_CONNECTED",
#endif
#ifdef CONFIG_TFL_USE_GATHER
    "GATHER",
#endif
#ifdef CONFIG_TFL_USE_GATHER_ND
    "GATHER_ND",
#endif
#ifdef CONFIG_TFL_USE_GREATER
    "GREATER",
#endif
#ifdef CONFIG_TFL_USE_GREATER_EQUAL
    "GREATER_EQUAL",
#endif
#ifdef CONFIG_TFL_USE_HARD_SWISH
    "HARD_SWISH",
#endif
#ifdef CONFIG_TFL_USE_IF
    "IF",
#endif
#ifdef CONFIG_TFL_USE_IRFFT
    "IRFFT",
#endif
#ifdef CONFIG_TFL_USE_L2_NORMALIZATION
    "L2_NORMALIZATION",
#endif
#ifdef CONFIG_TFL_USE_L2_POOL_2D
    "L2_POOL_2D",
#endif
#ifdef CONFIG_TFL_USE_LEAKY_RELU
    "LEAKY_RELU",
#endif
#ifdef CONFIG_TFL_USE_LESS
    "LESS",
#endif
#ifdef CONFIG_TFL_USE_LESS_EQUAL
    "LESS_EQUAL",
#endif
#ifdef CONFIG_TFL_USE_LOG
    "LOG",
#endif
#ifdef CONFIG_TFL_USE_LOGICAL_AND
    "LOGICAL_AND",
#endif
#ifdef CONFIG_TFL_USE_LOGICAL_NOT
    "LOGICAL_NOT",
#endif
#ifdef CONFIG_TFL_USE_LOGICAL_OR
    "LOGICAL_OR",
#endif
#ifdef CONFIG_TFL_USE_LOGISTIC
    "LOGISTIC",
#endif
#ifdef CONFIG_TFL_USE_LOG_SOFTMAX
    "LOG_SOFTMAX",
#endif
#ifdef CONFIG_TFL_USE_MAXIMUM
    "MAXIMUM",
#endif
#ifdef CONFIG_TFL_USE_MAX_POOL_2D
    "MAX_POOL_2D",
#endif
#ifdef CONFIG_TFL_USE_MIRROR_PAD
    "MIRROR_PAD",
#endif
#ifdef CONFIG_TFL_USE_MEAN
    "MEAN",
#endif
#ifdef CONFIG_TFL_USE_MINIMUM
    "MINIMUM",
#endif
#ifdef CONFIG_TFL_USE_MUL
    "MUL",
#endif
#ifdef CONFIG_TFL_USE_NEG
    "NEG",
#endif
#ifdef CONFIG_TFL_USE_NOT_EQUAL
    "NOT_EQUAL",
#endif
#ifdef CONFIG_TFL_USE_OVERLAP_ADD
    "OVERLAP_ADD",
#endif
#ifdef CONFIG_TFL_USE_PACK
    "PACK",
#endif
#ifdef CONFIG_TFL_USE_PAD
    "PAD",
#endif
#ifdef CONFIG_TFL_USE_PAD_V2
    "PAD_V2",
#endif
#ifdef CONFIG_TFL_USE_PCAN
    "PCAN",
#endif
#ifdef CONFIG_TFL_USE_PRELU
    "PRELU",
#endif
#ifdef CONFIG_TFL_USE_QUANTIZE
    "QUANTIZE",
#endif
#ifdef CONFIG_TFL_USE_READ_VARIABLE
    "READ_VARIABLE",
#endif
#ifdef CONFIG_TFL_USE_REDUCE_ALL
    "REDUCE_ALL",
#endif
#ifdef CONFIG_TFL_USE_REDUCE_MAX
    "REDUCE_MAX",
#endif
#ifdef CONFIG_TFL_USE_REDUCE_MIN
    "REDUCE_MIN",
#endif
#ifdef CONFIG_TFL_USE_RELU
    "RELU",
#endif
#ifdef CONFIG_TFL_USE_RELU6
    "RELU6",
#endif
#ifdef CONFIG_TFL_USE_RESHAPE
    "RESHAPE",
#endif
#ifdef CONFIG_TFL_USE_RESIZE_BILINEAR
    "RESIZE_BILINEAR",
#endif
#ifdef CONFIG_TFL_USE_RESIZE_NEAREST_NEIGHBOR
    "RESIZE_NEAREST_NEIGHBOR",
#endif
#ifdef CONFIG_TFL_USE_REVERSE_V2
    "REVERSE_V2",
#endif
#ifdef CONFIG_TFL_USE_RFFT
    "RFFT",
#endif
#ifdef CONFIG_TFL_USE_ROUND
    "ROUND",
#endif
#ifdef CONFIG_TFL_USE_RSQRT
    "RSQRT",
#endif
#ifdef CONFIG_TFL_USE_SELECT_V2
    "SELECT_V2",
#endif
#ifdef CONFIG_TFL_USE_SHAPE
    "SHAPE",
#endif
#ifdef CONFIG_TFL_USE_SIN
    "SIN",
#endif
#ifdef CONFIG_TFL_USE_SLICE
    "SLICE",
#endif
#ifdef CONFIG_TFL_USE_SOFTMAX
    "SOFTMAX",
#endif
#ifdef CONFIG_TFL_USE_SPACE_TO_BATCH_ND
    "SPACE_TO_BATCH_ND",
#endif
#ifdef CONFIG_TFL_USE_SPACE_TO_DEPTH
    "SPACE_TO_DEPTH",
#endif
#ifdef CONFIG_TFL_USE_SPLIT
    "SPLIT",
#endif
#ifdef CONFIG_TFL_USE_SPLIT_V
    "SPLIT_V",
#endif
#ifdef CONFIG_TFL_USE_SQUEEZE
    "SQUEEZE",
#endif
#ifdef CONFIG_TFL_USE_SQRT
    "SQRT",
#endif
#ifdef CONFIG_TFL_USE_SQUARE
    "SQUARE",
#endif
#ifdef CONFIG_TFL_USE_SQUARED_DIFFERENCE
    "SQUARED_DIFFERENCE",
#endif
#ifdef CONFIG_TFL_USE_STRIDED_SLICE
    "STRIDED_SLICE",
#endif
#ifdef CONFIG_TFL_USE_STACKER
    "STACKER",
#endif
#ifdef CONFIG_TFL_USE_SUB
    "SUB",
#endif
#ifdef CONFIG_TFL_USE_SUM
    "SUM",
#endif
#ifdef CONFIG_TFL_USE_SVDF
    "SVDF",
#endif
#ifdef CONFIG_TFL_USE_TANH
    "TANH",
#endif
#ifdef CONFIG_TFL_USE_TRANSPOSE_CONV
    "TRANSPOSE_CONV",
#endif
#ifdef CONFIG_TFL_USE_TRANSPOSE
    "TRANSPOSE",
#endif
#ifdef CONFIG_TFL_USE_UNPACK
    "UNPACK",
#endif
#ifdef CONFIG_TFL_USE_UNIDIRECTIONAL_SEQUENCE_LSTM
    "UNIDIRECTIONAL_SEQUENCE_LSTM",
#endif
#ifdef CONFIG_TFL_USE_VAR_HANDLE
    "VAR_HANDLE",
#endif
#ifdef CONFIG_TFL_USE_WHILE
    "WHILE",
#endif
#ifdef CONFIG_TFL_USE_WINDOW
    "WINDOW",
#endif
#ifdef CONFIG_TFL_USE_ZEROS_LIKE
    "ZEROS_LIKE",
#endif
};

const char** tflite_micro_get_enabled_ops(size_t* out_count) {
    if (out_count) {
        *out_count = sizeof(enabled_ops) / sizeof(enabled_ops[0]);
    }
    return enabled_ops;
}

} /* extern "C" */