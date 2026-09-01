#include "esp_fft_wrapper.h"
#include "dsps_fft2r.h"
#include "dsps_wind.h"
#include "esp_log.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char* TAG = "ESP_FFT_WRAPPER";
static bool s_fft_initialized = false;
static size_t s_max_fft_size = 0;

bool esp_fft_wrapper_init(size_t max_fft_size) {
    if (s_fft_initialized && max_fft_size <= s_max_fft_size) {
        return true;
    }

    esp_err_t err = dsps_fft2r_init_fc32(NULL, max_fft_size);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize dsps fft2r table (err: %d)", err);
        return false;
    }

    s_fft_initialized = true;
    s_max_fft_size = max_fft_size;
    return true;
}

bool esp_fft_wrapper_stft(const float* input, size_t input_len, size_t window_size, size_t hop_size, size_t fft_size, float* out_spectrogram) {
    if (!input || !out_spectrogram || window_size == 0 || hop_size == 0 || fft_size < window_size) {
        return false;
    }

    if (!esp_fft_wrapper_init(fft_size)) {
        return false;
    }

    size_t num_frames = (input_len - window_size) / hop_size + 1;
    size_t freq_bins = fft_size / 2;

    // Allocate temporary window and complex FFT buffers
    float* window = (float*)malloc(window_size * sizeof(float));
    float* fft_buffer = (float*)malloc(2 * fft_size * sizeof(float)); // complex interleaved
    if (!window || !fft_buffer) {
        free(window);
        free(fft_buffer);
        return false;
    }

    // Compute Hann window using ESP-DSP
    dsps_wind_hann_f32(window, window_size);

    for (size_t f = 0; f < num_frames; f++) {
        size_t start_idx = f * hop_size;

        // Prepare interleaved complex input: real = input * window, imag = 0
        for (size_t i = 0; i < fft_size; i++) {
            if (i < window_size) {
                fft_buffer[2 * i] = input[start_idx + i] * window[i];
            } else {
                fft_buffer[2 * i] = 0.0f;
            }
            fft_buffer[2 * i + 1] = 0.0f;
        }

        // Execute ESP-DSP radix-2 FFT
        dsps_fft2r_fc32(fft_buffer, fft_size);
        dsps_bit_rev_fc32(fft_buffer, fft_size);

        // Calculate magnitudes and store in spectrogram
        for (size_t k = 0; k < freq_bins; k++) {
            float re = fft_buffer[2 * k];
            float im = fft_buffer[2 * k + 1];
            out_spectrogram[f * freq_bins + k] = sqrtf(re * re + im * im);
        }
    }

    free(window);
    free(fft_buffer);
    return true;
}
