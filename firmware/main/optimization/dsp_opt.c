#include "dsp_opt.h"
#include "sdkconfig.h"

#if defined(CONFIG_IDF_TARGET_ESP32S3)

void dsp_opt_int16_to_float(const int16_t *src, float *dest, size_t len) {
    size_t i = 0;

    // Fast path: 4-way unrolled conversion with 2^-15 scaling
    for (; i + 3 < len; i += 4) {
        int32_t s0 = src[i + 0];
        int32_t s1 = src[i + 1];
        int32_t s2 = src[i + 2];
        int32_t s3 = src[i + 3];

        float f0, f1, f2, f3;

        asm volatile (
            "float.s %[f0], %[s0], 15 \n\t"
            "float.s %[f1], %[s1], 15 \n\t"
            "float.s %[f2], %[s2], 15 \n\t"
            "float.s %[f3], %[s3], 15 \n\t"
            : [f0] "=f" (f0), [f1] "=f" (f1), [f2] "=f" (f2), [f3] "=f" (f3)
            : [s0] "r" (s0), [s1] "r" (s1), [s2] "r" (s2), [s3] "r" (s3)
        );

        dest[i + 0] = f0;
        dest[i + 1] = f1;
        dest[i + 2] = f2;
        dest[i + 3] = f3;
    }

    // Scalar tail: uses the same float.s instruction for any remaining 1 to 3 elements
    for (; i < len; i++) {
        int32_t s = src[i];
        float f;

        asm volatile (
            "float.s %[f], %[s], 15 \n\t"
            : [f] "=f" (f)
            : [s] "r" (s)
        );

        dest[i] = f;
    }
}

#else

#include "dsps_math.h"

void dsp_opt_int16_to_float(const int16_t *src, float *dest, size_t len) {
    const float scale = 1.0f / 32768.0f;
    for (size_t i = 0; i < len; ++i) {
        dest[i] = (float)src[i];
    }
    // Multiply vector dest by scalar (hardware accelerated in esp-dsp)
    dsps_mulc_f32(dest, dest, len, scale, 1, 1);
}

#endif
