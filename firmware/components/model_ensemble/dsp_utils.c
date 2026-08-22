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
