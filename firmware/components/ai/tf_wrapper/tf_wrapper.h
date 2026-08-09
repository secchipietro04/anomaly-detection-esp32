#ifndef TFLITE_MICRO_WRAPPER_H
#define TFLITE_MICRO_WRAPPER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TFLITE_WRAPPER_OK = 0,
    TFLITE_WRAPPER_ERROR_INIT_MODEL,
    TFLITE_WRAPPER_ERROR_ALLOCATION,
    TFLITE_WRAPPER_ERROR_INTERPRETER,
    TFLITE_WRAPPER_ERROR_INVOKE,
    TFLITE_WRAPPER_ERROR_INVALID_PARAM
} tflite_status_t;

typedef struct {
    size_t tensor_arena_size;   /* Size of the arena in bytes */
    uint8_t* tensor_arena;      /* Pointer to external arena (NULL to auto-allocate) */
    bool use_psram;             /* Allocate arena in SPIRAM if auto-allocating */
} tflite_config_t;

typedef struct {
    void* data;                 /* Pointer to raw tensor data */
    size_t bytes;               /* Tensor buffer size in bytes */
    int type;                   /* TfLiteType enum numerical value */
    int dims[4];                /* Dimension array */
    size_t num_dims;            /* Number of active dimensions */
    float scale;                /* Quantization scale */
    int zero_point;             /* Quantization zero point */
} tflite_tensor_details_t;

typedef void* tflite_handle_t;

/**
 * @brief Initializes the TensorFlow Lite Micro interpreter.
 * @param model_buffer Pointer to flatbuffer model byte array.
 * @param config Pointer to initialization parameters.
 * @param status Optional output parameter for return status code.
 * @return Handle pointer to interpreter context, or NULL on failure.
 */
tflite_handle_t tflite_micro_init(const uint8_t* model_buffer, const tflite_config_t* config, tflite_status_t* status);

/**
 * @brief Executes model inference.
 */
tflite_status_t tflite_micro_invoke(tflite_handle_t handle);

/**
 * @brief Fetches input tensor metadata and underlying buffer pointer.
 */
tflite_status_t tflite_micro_get_input_details(tflite_handle_t handle, size_t index, tflite_tensor_details_t* details);

/**
 * @brief Fetches output tensor metadata and underlying buffer pointer.
 */
tflite_status_t tflite_micro_get_output_details(tflite_handle_t handle, size_t index, tflite_tensor_details_t* details);

/**
 * @brief Frees interpreter context and internally allocated memory.
 */
void tflite_micro_free(tflite_handle_t handle);

/**
 * @brief Returns the list of enabled custom operations (TFLite adds) registered at build-time.
 * @param out_count Output pointer to receive the size of the returned string array.
 * @return Array of pointers to static strings containing the operator names.
 */
const char** tflite_micro_get_enabled_ops(size_t* out_count);

#ifdef __cplusplus
}
#endif

#endif /* TFLITE_MICRO_WRAPPER_H */