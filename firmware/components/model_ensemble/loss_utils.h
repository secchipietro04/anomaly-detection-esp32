#ifndef LOSS_UTILS_H
#define LOSS_UTILS_H

#include "base_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Computes reconstruction loss between target and reconstructed slices.
 * @param target Pointer to target float slice.
 * @param reconstruction Pointer to reconstructed float slice.
 * @param bins Number of elements in the slices.
 * @param loss_mode Loss computation contract (LogMSE, LinearMSE, etc.).
 * @return Computed loss value.
 */
float compute_reconstruction_loss(const float* target, const float* reconstruction, uint32_t bins, LossMode_t loss_mode);

#ifdef __cplusplus
}
#endif

#endif // LOSS_UTILS_H
