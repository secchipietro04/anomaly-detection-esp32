#ifndef BASE_MODEL_H
#define BASE_MODEL_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "tf_wrapper.h"
#include "anomaly_model_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t arena_size;          /* Size of tensor arena in bytes */
    uint8_t* shared_arena;      /* Pointer to shared buffer. If NULL, memory is allocated on heap/PSRAM */
    bool use_psram;             /* Allocate arena in SPIRAM if shared_arena is NULL */
} base_model_config_t;

typedef struct {
    tflite_handle_t tf_handle;  /* Opaque handle to TFLite interpreter */
    uint8_t* arena_ptr;         /* Active tensor arena buffer pointer */
    size_t arena_size;          /* Active arena size */
    bool arena_owned;           /* True if arena was dynamically allocated by base_model */
    tflite_tensor_details_t input_details;
    tflite_tensor_details_t output_details;
} base_model_t;

/**
 * @brief Full initialization: sets up arena and loads model interpreter.
 */
bool base_model_init(base_model_t* model, const uint8_t* model_bytes, const base_model_config_t* config);

/**
 * @brief Frees the interpreter context but keeps the arena buffer allocated in memory.
 */
void base_model_unload_interpreter(base_model_t* model);

/**
 * @brief Reloads a model (same or different binary) into the existing tensor arena.
 */
bool base_model_reload(base_model_t* model, const uint8_t* new_model_bytes);

/**
 * @brief Deallocates the arena buffer if it was internally owned.
 */
void base_model_free_arena(base_model_t* model);

/**
 * @brief Unloads interpreter and frees arena (full cleanup).
 */
void base_model_deinit(base_model_t* model);

/**
 * @brief Executes inference using the current input tensor buffer.
 */
bool base_model_run(base_model_t* model);

#ifdef __cplusplus
}
#endif

#endif /* BASE_MODEL_H */