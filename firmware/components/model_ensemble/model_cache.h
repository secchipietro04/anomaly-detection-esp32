#ifndef MODEL_CACHE_H
#define MODEL_CACHE_H

#include "base_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_CACHE_CAPACITY 16

typedef struct {
    ModelInstance_t model;
} cached_model_t;

typedef struct {
    cached_model_t items[MAX_CACHE_CAPACITY];
    uint32_t count;
    uint32_t capacity;
    size_t arena_size;
} model_cache_t;

/**
 * @brief Initialize the model cache.
 */
void model_cache_init(model_cache_t* cache, uint32_t capacity, size_t arena_size);

/**
 * @brief Clear and deinitialize all models in the cache.
 */
void model_cache_clear(model_cache_t* cache);

/**
 * @brief Retrieve a model from the cache by its ID.
 *        If found, moves the model to the front of the cache (index 0).
 * @return Pointer to ModelInstance_t, or NULL if not found.
 */
ModelInstance_t* model_cache_get(model_cache_t* cache, uint32_t model_id);

/**
 * @brief Insert a new model into the cache.
 *        If the cache is full, evicts the model at the bottom (index count-1).
 *        If memory allocation fails, returns MODEL_ERR_NO_MEM.
 * @return MODEL_SUCCESS on success, error code on failure.
 */
int model_cache_insert(model_cache_t* cache, const uint8_t* model_data, size_t model_size);

/**
 * @brief Change the cache capacity.
 *        If the new capacity is smaller than current count, evicts oldest models.
 */
void model_cache_set_capacity(model_cache_t* cache, uint32_t new_capacity);

#ifdef __cplusplus
}
#endif

#endif // MODEL_CACHE_H
