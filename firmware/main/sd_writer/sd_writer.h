#ifndef SD_WRITER_H
#define SD_WRITER_H

#include "quad_buffer.h"
#include <stdint.h>

// base unpadded binary record struct
typedef struct {
    uint32_t segment_id;
    float sample_rate;
    float accel_x[4096];
    float accel_y[4096];
    float accel_z[4096];
    float gyro_x[4096];
    float gyro_y[4096];
    float gyro_z[4096];
    uint8_t has_inference;
    uint32_t router_model_id;
    uint32_t memory_model_id;
    uint32_t active_submodel_id;
    float anomaly_score;
    uint8_t is_anomaly;
    uint8_t was_sent;
} __attribute__((packed)) sd_record_data_t;

// compile time sector alignment math
#define RECORD_SECTORS         (((sizeof(sd_record_data_t)) + 511) / 512)
#define RECORD_SIZE            (RECORD_SECTORS * 512)
#define RECORD_PADDING         (RECORD_SIZE - sizeof(sd_record_data_t))

typedef struct {
    sd_record_data_t data;
    uint8_t padding[RECORD_PADDING];
} __attribute__((packed)) sd_record_t;

// start the sd writer task
void sd_writer_start(quad_buffer_t *qb);

#endif // SD_WRITER_H
