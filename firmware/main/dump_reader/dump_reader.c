#include "dump_reader.h"
#include "sd_writer.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char *TAG = "dump_reader";

// globals
extern TaskHandle_t harvester_task_handle;
extern TaskHandle_t inferencer_task_handle;
extern TaskHandle_t uploader_task_handle;

extern uint32_t global_dump_time_ms;
extern bool global_dump_reading;
extern float global_dump_ips;
extern void publish_health_info(void); // Defined in main_functions



typedef struct {
    quad_buffer_t *qb;
    bool run_inference;
} dump_task_args_t;

static void dump_reader_task(void *pvParameters) {
    dump_task_args_t *args = (dump_task_args_t *)pvParameters;
    quad_buffer_t *qb = args->qb;
    bool run_inference = args->run_inference;
    
    ESP_LOGI(TAG, "historical dump task started (inference: %d)", run_inference);
    
    // Suspend real-time harvester thread
    if (harvester_task_handle) {
        vTaskSuspend(harvester_task_handle);
        ESP_LOGI(TAG, "suspended harvester task for dump replay");
    }
    // drain any in-flight real-time harvester segments before starting the timer
    int prep_slots[4];
    for (int i = 0; i < 4; i++) {
        xQueueReceive(free_slots_queue, &prep_slots[i], portMAX_DELAY);
    }
    
    global_dump_reading = true;
    uint32_t total_inferences = 0;
    
    // record start time when the pipeline is completely clean
    int64_t start_time = esp_timer_get_time();
    
    // return slots so that the dump scan loop can use them
    for (int i = 0; i < 4; i++) {
        xQueueSend(free_slots_queue, &prep_slots[i], 0);
    }
    
    sd_record_t *rec = malloc(sizeof(sd_record_t));
    if (!rec) {
        ESP_LOGE(TAG, "OOM allocating record memory");
        if (harvester_task_handle) vTaskResume(harvester_task_handle);
        global_dump_reading = false;
        free(args);
        vTaskDelete(NULL);
        return;
    }
    
    // loop through the 64 data files
    for (int file_idx = 0; file_idx < 64; file_idx++) {
        char filepath[64];
        snprintf(filepath, sizeof(filepath), "/sdcard/data/data_%d.bin", file_idx);
        
        struct stat st;
        if (stat(filepath, &st) != 0) {
            // file doesn't exist, skip
            continue;
        }
        
        FILE *f = fopen(filepath, "rb");
        if (!f) continue;
        
        ESP_LOGI(TAG, "replaying file %s", filepath);
        
        while (fread(rec, 1, RECORD_SIZE, f) == RECORD_SIZE) {
            // skip replaying this segment if it was already sent in real-time
            if (rec->data.was_sent) {
                continue;
            }
            
            // wait for a free slot from the queue
            int free_slot_idx;
            if (xQueueReceive(free_slots_queue, &free_slot_idx, portMAX_DELAY) != pdTRUE) {
                continue;
            }
            buffer_slot_t *slot = &qb->slots[free_slot_idx];
            
            // lock and load
            quad_buffer_lock(slot);
            slot->segment_id = rec->data.segment_id;
            slot->sample_rate = rec->data.sample_rate;
            slot->is_replay = true;
            slot->is_inferenced = false;
            slot->is_anomaly = false;
            
            memcpy(slot->data.accel_x, rec->data.accel_x, sizeof(float) * 4096);
            memcpy(slot->data.accel_y, rec->data.accel_y, sizeof(float) * 4096);
            memcpy(slot->data.accel_z, rec->data.accel_z, sizeof(float) * 4096);
            memcpy(slot->data.gyro_x, rec->data.gyro_x, sizeof(float) * 4096);
            memcpy(slot->data.gyro_y, rec->data.gyro_y, sizeof(float) * 4096);
            memcpy(slot->data.gyro_z, rec->data.gyro_z, sizeof(float) * 4096);
            

            
            // set flags and notify tasks via the event queues
            if (run_inference) {
                slot->flags = BUF_FLAG_READY_INF | BUF_FLAG_FORCE_INF;
                quad_buffer_unlock(slot);
                
                xQueueSend(ready_inf_queue, &free_slot_idx, 0);
                total_inferences += 1;
            } else {
                slot->flags = BUF_FLAG_READY_SEND;
                quad_buffer_unlock(slot);
                
                xQueueSend(ready_send_queue, &free_slot_idx, 0);
            }
            
            // limit rate slightly to avoid overwhelming tasks
            vTaskDelay(pdMS_TO_TICKS(50));
        }
        
        fclose(f);
    }
    
    free(rec);
    ESP_LOGI(TAG, "done scanning SD card files. waiting for pending buffers to clear...");
    
    // block until uploader and inferencer complete all pending replayed buffers (draining the pipeline)
    int drained_slots[4];
    for (int i = 0; i < 4; i++) {
        xQueueReceive(free_slots_queue, &drained_slots[i], portMAX_DELAY);
    }
    
    // calculate statistics
    int64_t end_time = esp_timer_get_time();
    
    // return the 4 slots back to the free queue so that harvester can start using them again
    for (int i = 0; i < 4; i++) {
        xQueueSend(free_slots_queue, &drained_slots[i], 0);
    }
    uint32_t diff_ms = (uint32_t)((end_time - start_time) / 1000);
    global_dump_time_ms = diff_ms;
    global_dump_reading = false;
    
    if (run_inference && diff_ms > 0) {
        global_dump_ips = (float)total_inferences / ((float)diff_ms / 1000.0f);
    } else {
        global_dump_ips = 0.0f;
    }
    
    ESP_LOGI(TAG, "dump completed Time: %u ms, IPS: %f", (unsigned int)diff_ms, global_dump_ips);
    
    // send telemetry update immediately with the final statistics
    publish_health_info();
    
    // Resume real-time harvester thread
    if (harvester_task_handle) {
        vTaskResume(harvester_task_handle);
        ESP_LOGI(TAG, "resumed harvester task");
    }
    
    free(args);
    vTaskDelete(NULL);
}

void dump_reader_trigger(quad_buffer_t *qb, bool run_inference) {
    dump_task_args_t *args = malloc(sizeof(dump_task_args_t));
    if (args) {
        args->qb = qb;
        args->run_inference = run_inference;
        
        xTaskCreatePinnedToCore(
            dump_reader_task,
            "dump_reader",
            8192,
            args,
            3,
            NULL,
            0
        );
    }
}
