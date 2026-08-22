#ifndef INFERENCER_H
#define INFERENCER_H

#include "quad_buffer.h"

#define FFT_WINDOW_SIZE         256
#define FFT_HOP_SIZE            64
#define FFT_SIZE                256

// Mathematically derived spectrogram shape
#define SPECTROGRAM_BINS        ((FFT_SIZE / 2) + 1)                                            // 129
#define SPECTROGRAM_FRAMES      (((SAMPLES_PER_SEGMENT - FFT_WINDOW_SIZE) / FFT_HOP_SIZE) + 1) // 61

// start inferencer task
void inferencer_start(quad_buffer_t *qb);

#endif // INFERENCER_H
