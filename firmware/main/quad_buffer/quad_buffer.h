#ifndef QUAD_BUFFER_H
#define QUAD_BUFFER_H

#include <stdint.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#define BUF_FLAG_FREE        0
#define BUF_FLAG_HARVESTING  (1 << 0)
#define BUF_FLAG_READY_INF   (1 << 1)
#define BUF_FLAG_READY_SEND  (1 << 2)
#define BUF_FLAG_PENDING_SD  (1 << 3)
#define BUF_FLAG_FORCE_INF   (1 << 4)

#define SAMPLES_PER_SEGMENT  4096

typedef struct {
    float accel_x[SAMPLES_PER_SEGMENT];
    float accel_y[SAMPLES_PER_SEGMENT];
    float accel_z[SAMPLES_PER_SEGMENT];
    float gyro_x[SAMPLES_PER_SEGMENT];
    float gyro_y[SAMPLES_PER_SEGMENT];
    float gyro_z[SAMPLES_PER_SEGMENT];
} sensor_data_t;

typedef struct {
    sensor_data_t data;
    uint32_t flags;
    uint32_t segment_id;
    float sample_rate;
    SemaphoreHandle_t mutex; // lock for state change
    
    // results from model
    uint32_t router_model_id;
    uint32_t memory_model_id;
    uint32_t active_submodel_id;
    float anomaly_score;
    bool is_anomaly;
    bool is_inferenced;
    bool is_replay; // true if block is read from SD card for dump
#if defined(CONFIG_RECORD_INFERENCE_TIME) || defined(RECORD_INFERENCE_TIME)
    uint32_t inference_time_ms;
#endif
} buffer_slot_t;

typedef struct {
    buffer_slot_t slots[4];
    uint8_t write_idx;
    uint8_t process_idx;
} quad_buffer_t;

// Init the quad buffer structure and the slot mutexes
#include "freertos/queue.h"
extern QueueHandle_t free_slots_queue;
extern QueueHandle_t ready_inf_queue;
extern QueueHandle_t pending_sd_queue;
extern QueueHandle_t ready_send_queue;

void quad_buffer_init(quad_buffer_t *qb);

// lock and unlock slot flags
void quad_buffer_lock(buffer_slot_t *slot);
void quad_buffer_unlock(buffer_slot_t *slot);

#endif // QUAD_BUFFER_H
