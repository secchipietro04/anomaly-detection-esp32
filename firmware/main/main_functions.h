#ifndef MAIN_FUNCTIONS_H
#define MAIN_FUNCTIONS_H

#include <stdint.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#ifdef __cplusplus
extern "C" {
#endif

// global config values
extern float global_sample_rate;
extern bool global_sd_enabled;
extern bool global_inference_enabled;
extern uint32_t global_stream_mode;
extern uint32_t global_cadence;
extern uint32_t global_heartbeat_s;

// dump / stats globals
extern uint32_t global_dump_time_ms;
extern bool global_dump_reading;
extern float global_dump_ips;

// task handles
extern TaskHandle_t harvester_task_handle;
extern TaskHandle_t inferencer_task_handle;
extern TaskHandle_t sd_writer_task_handle;
extern TaskHandle_t uploader_task_handle;

// hardware node identifier
extern char global_node_id[32];

void setup(void);
void loop(void);

// send current health telemetry packet
void publish_health_info(void);

#ifdef __cplusplus
}
#endif

#endif // MAIN_FUNCTIONS_H
