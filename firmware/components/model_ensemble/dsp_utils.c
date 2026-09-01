#include "dsp_utils.h"

void pool_1d_max_pow2(const float* in, size_t in_len, float* out, size_t out_len) {
    if (!in || !out || out_len == 0 || in_len < out_len) return;
    size_t K = in_len / out_len;
    for (size_t i = 0; i < out_len; i++) {
        float max_val = in[i * K];
        for (size_t k = 1; k < K; k++) {
            float val = in[i * K + k];
            if (val > max_val) {
                max_val = val;
            }
        }
        out[i] = max_val;
    }
}

void dsp_fast_log1p_vec(const float* src, float* dst, size_t len) {
    if (!src || !dst || len == 0) return;
    
    // Unroll 4x for dual-issue FPU
    size_t i = 0;
    for (; i + 3 < len; i += 4) {
        float x0 = src[i] > 0.0f ? src[i] : 0.0f;
        float x1 = src[i+1] > 0.0f ? src[i+1] : 0.0f;
        float x2 = src[i+2] > 0.0f ? src[i+2] : 0.0f;
        float x3 = src[i+3] > 0.0f ? src[i+3] : 0.0f;

        union { float f; unsigned int i; } u0 = { .f = 1.0f + x0 };
        union { float f; unsigned int i; } u1 = { .f = 1.0f + x1 };
        union { float f; unsigned int i; } u2 = { .f = 1.0f + x2 };
        union { float f; unsigned int i; } u3 = { .f = 1.0f + x3 };

        dst[i]   = ((float)u0.i * 1.1920928955e-7f - 127.0f) * 0.69314718056f;
        dst[i+1] = ((float)u1.i * 1.1920928955e-7f - 127.0f) * 0.69314718056f;
        dst[i+2] = ((float)u2.i * 1.1920928955e-7f - 127.0f) * 0.69314718056f;
        dst[i+3] = ((float)u3.i * 1.1920928955e-7f - 127.0f) * 0.69314718056f;
    }
    for (; i < len; i++) {
        float x = src[i] > 0.0f ? src[i] : 0.0f;
        union { float f; unsigned int i; } u = { .f = 1.0f + x };
        dst[i] = ((float)u.i * 1.1920928955e-7f - 127.0f) * 0.69314718056f;
    }
}
