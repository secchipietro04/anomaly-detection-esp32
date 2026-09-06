#include "model_cache.h"
#include <string.h>

void model_cache_init(model_cache_t* cache, uint32_t capacity, size_t arena_size) {
    if (!cache) return;
    memset(cache, 0, sizeof(model_cache_t));
    cache->capacity = capacity > MAX_CACHE_CAPACITY ? MAX_CACHE_CAPACITY : capacity;
    cache->arena_size = arena_size;
}

void model_cache_clear(model_cache_t* cache) {
    if (!cache) return;
    for (uint32_t i = 0; i < cache->count; i++) {
        model_instance_deinit(&cache->items[i].model);
    }
    cache->count = 0;
}

ModelInstance_t* model_cache_get(model_cache_t* cache, uint32_t model_id) {
    if (!cache) return NULL;
    for (uint32_t i = 0; i < cache->count; i++) {
        if (cache->items[i].model.config.model_id == model_id) {
            // Shift target item to index 0 (Most Recently Used / LIFO top)
            cached_model_t target = cache->items[i];
            for (uint32_t j = i; j > 0; j--) {
                cache->items[j] = cache->items[j - 1];
            }
            cache->items[0] = target;
            return &cache->items[0].model;
        }
    }
    return NULL;
}

int model_cache_insert(model_cache_t* cache, const uint8_t* model_data, size_t model_size) {
    if (!cache || !model_data || model_size == 0) return MODEL_ERR_GENERIC;

    ModelInstance_t new_model;
    memset(&new_model, 0, sizeof(ModelInstance_t));

    // Initialize the model in a temporary instance first (auto-shrink if out of memory)
    int init_ret = model_instance_init(&new_model, model_data, model_size, cache->arena_size);
    while (init_ret == MODEL_ERR_NO_MEM && cache->count > 0) {
        // Evict oldest cached model to free heap memory for the new model
        model_instance_deinit(&cache->items[cache->count - 1].model);
        cache->count--;
        init_ret = model_instance_init(&new_model, model_data, model_size, cache->arena_size);
    }
    if (init_ret != MODEL_SUCCESS) {
        return init_ret;
    }

    if (cache->capacity == 0) {
        model_instance_deinit(&new_model);
        return MODEL_ERR_GENERIC;
    }

    // Evict oldest if capacity is reached
    if (cache->count == cache->capacity) {
        model_instance_deinit(&cache->items[cache->count - 1].model);
        for (uint32_t i = cache->count - 1; i > 0; i--) {
            cache->items[i] = cache->items[i - 1];
        }
    } else {
        for (uint32_t i = cache->count; i > 0; i--) {
            cache->items[i] = cache->items[i - 1];
        }
        cache->count++;
    }

    cache->items[0].model = new_model;
    return MODEL_SUCCESS;
}

void model_cache_set_capacity(model_cache_t* cache, uint32_t new_capacity) {
    if (!cache) return;
    uint32_t target_cap = new_capacity > MAX_CACHE_CAPACITY ? MAX_CACHE_CAPACITY : new_capacity;

    if (target_cap < cache->count) {
        for (uint32_t i = target_cap; i < cache->count; i++) {
            model_instance_deinit(&cache->items[i].model);
        }
        cache->count = target_cap;
    }
    cache->capacity = target_cap;
}
