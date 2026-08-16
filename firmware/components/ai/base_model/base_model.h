#ifndef BASE_MODEL_H
#define BASE_MODEL_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MODEL_SUCCESS            0
#define MODEL_ERR_GENERIC       -1
#define MODEL_ERR_SCHEMA        -2
#define MODEL_ERR_NO_MEM        -3
#define MODEL_ERR_TFLM          -4

typedef enum {
    ARCHETYPE_AUTOENCODER = 1,
    ARCHETYPE_ROUTER = 2,
    ARCHETYPE_MEMORY = 3
} ModelArchetype_t;

typedef enum {
    LOSS_MODE_LOG_MSE = 1,
    LOSS_MODE_LINEAR_MSE = 2
} LossMode_t;

typedef struct {
    uint32_t model_id;
    ModelArchetype_t archetype;
    uint32_t temporal_depth;      // Number of chronological FFT slices
    uint32_t frequency_bins;     // Number of frequency channels per slice
    uint32_t d;                   // State dimension
    LossMode_t loss_mode;         // LogMSE or LinearMSE
    float anomaly_threshold;      // Threshold for autoencoders
    uint32_t route_mapping[16];   // Route mapping index -> submodel_id
    uint32_t num_modes;           // Number of output logits for router
} ModelConfig_t;

#ifdef __cplusplus
namespace tflite {
    class MicroInterpreter;
}
typedef struct ModelInstance {
    ModelConfig_t config;
    const uint8_t* model_data;  // Self-owned dynamically allocated copy of flatbuffer binary
    tflite::MicroInterpreter* interpreter;
    uint8_t* tensor_arena;
    size_t arena_size;
    
    int (*init)(struct ModelInstance* self, const uint8_t* model_data, size_t model_size, size_t arena_size);
    void (*deinit)(struct ModelInstance* self);
    void (*invoke)(struct ModelInstance* self);
} ModelInstance_t;
#else
typedef struct ModelInstance {
    ModelConfig_t config;
    const uint8_t* model_data;  // Self-owned dynamically allocated copy of flatbuffer binary
    void* interpreter;
    uint8_t* tensor_arena;
    size_t arena_size;
    
    int (*init)(struct ModelInstance* self, const uint8_t* model_data, size_t model_size, size_t arena_size);
    void (*deinit)(struct ModelInstance* self);
    void (*invoke)(struct ModelInstance* self);
} ModelInstance_t;
#endif

/**
 * @brief Initialize a model instance with its TFLite model data, creating a local copy.
 * @param self Pointer to the ModelInstance struct.
 * @param model_data Source flatbuffer model binary byte array.
 * @param model_size Size of the model binary in bytes.
 * @param arena_size Pre-allocated arena size.
 * @return 0 on success, negative value on failure.
 */
int model_instance_init(ModelInstance_t* self, const uint8_t* model_data, size_t model_size, size_t arena_size);

/**
 * @brief Deinitialize a model instance and free all its owned resources (including the local model data copy).
 * @param self Pointer to the ModelInstance struct.
 */
void model_instance_deinit(ModelInstance_t* self);

/**
 * @brief Execute model inference once.
 * @param self Pointer to the ModelInstance struct.
 */
void model_instance_invoke(ModelInstance_t* self);

/**
 * @brief Get the float input tensor buffer pointer for a model instance.
 * @param self Pointer to the ModelInstance struct.
 * @param index Input tensor index.
 * @return Pointer to input float data, or NULL on error.
 */
float* model_instance_get_input_tensor(ModelInstance_t* self, size_t index);

/**
 * @brief Get the float output tensor buffer pointer for a model instance.
 * @param self Pointer to the ModelInstance struct.
 * @param index Output tensor index.
 * @return Const pointer to output float data, or NULL on error.
 */
const float* model_instance_get_output_tensor(ModelInstance_t* self, size_t index);

/**
 * @brief Get the size of the input tensor in float elements.
 * @param self Pointer to the ModelInstance struct.
 * @param index Input tensor index.
 * @return Number of float elements in the input tensor.
 */
size_t model_instance_get_input_size(ModelInstance_t* self, size_t index);

/**
 * @brief Get the size of the output tensor in float elements.
 * @param self Pointer to the ModelInstance struct.
 * @param index Output tensor index.
 * @return Number of float elements in the output tensor.
 */
size_t model_instance_get_output_size(ModelInstance_t* self, size_t index);

#ifdef __cplusplus
}
#endif

#endif /* BASE_MODEL_H */