#ifndef DSP_OPT_H
#define DSP_OPT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Converts an array of 16-bit signed integers to 32-bit floating point numbers,
 *        scaling the values by 1/32768.0f (Q15 mapping to [-1.0f, 1.0f]).
 *        Uses assembly optimization on ESP32-S3 and falls back to ESP-DSP multiplication on other targets.
 * 
 * @param src Input pointer to 16-bit signed integer buffer
 * @param dest Output pointer to 32-bit floating point buffer
 * @param len Number of elements to convert
 */
void dsp_opt_int16_to_float(const int16_t *src, float *dest, size_t len);

#ifdef __cplusplus
}
#endif

#endif // DSP_OPT_H
