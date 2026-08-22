#include "ring_buffer.h"
#include "dsp_utils.h"
#include <stdlib.h>
#include <string.h>

bool ring_buffer_init(ring_buffer_t* rb, uint32_t depth, uint32_t stride) {
    if (!rb || depth == 0 || stride == 0) return false;
    memset(rb, 0, sizeof(ring_buffer_t));

    rb->data = (float*)calloc(depth * stride, sizeof(float));
    if (!rb->data) return false;

    rb->model_ids = (uint32_t*)calloc(depth, sizeof(uint32_t));
    if (!rb->model_ids) {
        free(rb->data);
        rb->data = NULL;
        return false;
    }

    rb->depth = depth;
    rb->stride = stride;
    return true;
}

void ring_buffer_free(ring_buffer_t* rb) {
    if (!rb) return;
    if (rb->data) {
        free(rb->data);
        rb->data = NULL;
    }
    if (rb->model_ids) {
        free(rb->model_ids);
        rb->model_ids = NULL;
    }
    rb->head = 0;
}

void ring_buffer_clear(ring_buffer_t* rb) {
    if (!rb) return;
    if (rb->data) {
        memset(rb->data, 0, rb->depth * rb->stride * sizeof(float));
    }
    if (rb->model_ids) {
        memset(rb->model_ids, 0, rb->depth * sizeof(uint32_t));
    }
    rb->head = 0;
}

void ring_buffer_write(ring_buffer_t* rb, const float* slice, uint32_t model_id) {
    if (!rb || !rb->data) return;
    memcpy(&rb->data[rb->head * rb->stride], slice, rb->stride * sizeof(float));
    rb->model_ids[rb->head] = model_id;
    rb->head = (rb->head + 1) % rb->depth;
}

void ring_buffer_set_last_model_id(ring_buffer_t* rb, uint32_t model_id) {
    if (!rb || !rb->model_ids) return;
    int latest_idx = ((int)rb->head - 1 + (int)rb->depth) % (int)rb->depth;
    rb->model_ids[latest_idx] = model_id;
}

const float* ring_buffer_get_slice(const ring_buffer_t* rb, int offset) {
    if (!rb || !rb->data) return NULL;
    int idx = ((int)rb->head + offset) % (int)rb->depth;
    if (idx < 0) {
        idx += rb->depth;
    }
    return &rb->data[idx * rb->stride];
}

void ring_buffer_unroll(const ring_buffer_t* rb, float* target, uint32_t target_C, uint32_t T) {
    if (!rb || !target || T == 0) return;
    for (uint32_t t = 0; t < T; t++) {
        int offset = -(int)T + (int)t;
        const float* src_slice = ring_buffer_get_slice(rb, offset);
        float* dest_ptr = target + (t * target_C);
        if (target_C < rb->stride) {
            pool_1d_max_pow2(src_slice, rb->stride, dest_ptr, target_C);
        } else {
            memcpy(dest_ptr, src_slice, target_C * sizeof(float));
        }
    }
}

bool ring_buffer_verify_sequence(const ring_buffer_t* rb, uint32_t target_id, uint32_t T) {
    if (!rb || !rb->model_ids || T == 0) return false;
    for (uint32_t t = 0; t < T; t++) {
        int offset = -(int)T + (int)t;
        int idx = ((int)rb->head + offset) % (int)rb->depth;
        if (idx < 0) {
            idx += rb->depth;
        }
        if (rb->model_ids[idx] != target_id) {
            return false;
        }
    }
    return true;
}
