#ifndef ESP_FFT_WRAPPER_H
#define ESP_FFT_WRAPPER_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the ESP-DSP FFT tables.
 * @param max_fft_size Maximum FFT size supported (e.g. 1024).
 * @return true on success, false on failure.
 */
bool esp_fft_wrapper_init(size_t max_fft_size);

/**
 * @brief Compute STFT using ESP-DSP optimized functions.
 * @param input Input raw signal data.
 * @param input_len Length of raw signal data.
 * @param window_size Size of the analysis window.
 * @param hop_size Step size between frames.
 * @param fft_size Size of FFT (must be power of 2 >= window_size).
 * @param out_spectrogram Output magnitude spectrogram buffer of size: num_frames * (fft_size/2 + 1).
 * @return true on success, false on failure.
 */
bool esp_fft_wrapper_stft(const float* input, size_t input_len, size_t window_size, size_t hop_size, size_t fft_size, float* out_spectrogram);

#ifdef __cplusplus
}
#endif

#endif // ESP_FFT_WRAPPER_H
