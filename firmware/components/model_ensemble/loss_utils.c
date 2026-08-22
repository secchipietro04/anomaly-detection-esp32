#include "loss_utils.h"
#include <math.h>

float compute_reconstruction_loss(const float* target, const float* reconstruction, uint32_t bins, LossMode_t loss_mode) {
    if (!target || !reconstruction || bins == 0) {
        return 0.0f;
    }

    float loss_sum = 0.0f;
    switch (loss_mode) {
        case LOSS_MODE_LOG_MSE:
            for (uint32_t i = 0; i < bins; i++) {
                float diff = target[i] - reconstruction[i];
                loss_sum += diff * diff;
            }
            break;

        case LOSS_MODE_LINEAR_MSE:
            for (uint32_t i = 0; i < bins; i++) {
                float lin_target = expm1f(target[i]);
                float lin_recon = expm1f(reconstruction[i]);
                float diff = lin_target - lin_recon;
                loss_sum += diff * diff;
            }
            break;

        default:
            // Fallback to standard LogMSE
            for (uint32_t i = 0; i < bins; i++) {
                float diff = target[i] - reconstruction[i];
                loss_sum += diff * diff;
            }
            break;
    }

    return loss_sum / (float)bins;
}
