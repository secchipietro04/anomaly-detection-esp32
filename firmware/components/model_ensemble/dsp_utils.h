#ifndef DSP_UTILS_H
#define DSP_UTILS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Downsamples a frequency buffer using 1D max pooling.
 * @param in Pointer to source float buffer.
 * @param in_len Input length (must be power of 2).
 * @param out Pointer to destination float buffer.
 * @param out_len Target length (must be power of 2, <= in_len).
 */
void pool_1d_max_pow2(const float* in, size_t in_len, float* out, size_t out_len);

/**
 * @brief vectorized log1p ln(1+x) transform
 */
void dsp_fast_log1p_vec(const float* src, float* dst, size_t len);

#ifdef __cplusplus
}
#endif

#endif // DSP_UTILS_H
