#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float* data;                   // Size: depth * stride floats
    uint32_t* model_ids;           // Size: depth integers
    uint32_t head;                 // Circular write index
    uint32_t depth;                // Total slices capacity
    uint32_t stride;               // Dimension width of each slice (C)
} ring_buffer_t;

/**
 * @brief Initialize the circular ring buffer.
 * @return true on success, false on allocation failure.
 */
bool ring_buffer_init(ring_buffer_t* rb, uint32_t depth, uint32_t stride);

/**
 * @brief Free ring buffer memory.
 */
void ring_buffer_free(ring_buffer_t* rb);

/**
 * @brief Zero out all data and reset the head index.
 */
void ring_buffer_clear(ring_buffer_t* rb);

/**
 * @brief Write a slice and its routed model ID into the ring buffer.
 */
void ring_buffer_write(ring_buffer_t* rb, const float* slice, uint32_t model_id);

/**
 * @brief Directly update the model ID of the last written slice.
 */
void ring_buffer_set_last_model_id(ring_buffer_t* rb, uint32_t model_id);

/**
 * @brief Get the pointer to a slice at a relative chronological index.
 * @param offset Relative index: -1 is the latest, -depth is the oldest.
 */
const float* ring_buffer_get_slice(const ring_buffer_t* rb, int offset);

/**
 * @brief Unrolls T chronological slices into a contiguous destination array.
 *        Applies downsampling automatically if target_C < stride.
 */
void ring_buffer_unroll(const ring_buffer_t* rb, float* target, uint32_t target_C, uint32_t T);

/**
 * @brief Verifies if all the last T slices were routed to the target model ID.
 */
bool ring_buffer_verify_sequence(const ring_buffer_t* rb, uint32_t target_id, uint32_t T);

#ifdef __cplusplus
}
#endif

#endif // RING_BUFFER_H
