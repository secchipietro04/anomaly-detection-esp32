#include "sd_writer.h"
#include "esp_log.h"
#include "freertos/task.h"
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/unistd.h>

static const char *TAG = "sd_writer";

#define MAX_FILES 64

// capacity limit for 16MB file
#define MAX_RECORDS_PER_FILE   ((16 * 1024 * 1024) / RECORD_SIZE)

typedef struct {
    uint32_t active_file_idx;
    uint32_t record_count;
} sd_state_t;

extern TaskHandle_t sd_writer_task_handle;
extern bool global_sd_enabled;

// active file tracker state
static sd_state_t state = {0, 0};

static void load_state(void) {
    FILE *f = fopen("/sdcard/data/state.bin", "rb");
    if (f) {
        fread(&state, sizeof(sd_state_t), 1, f);
        fclose(f);
        ESP_LOGI(TAG, "loaded SD state: active file %u, records %u", 
                 (unsigned int)state.active_file_idx, (unsigned int)state.record_count);
    } else {
        ESP_LOGI(TAG, "no SD state file found. using defaults.");
    }
}

static void save_state(void) {
    FILE *f = fopen("/sdcard/data/state.bin", "wb");
    if (f) {
        fwrite(&state, sizeof(sd_state_t), 1, f);
        fclose(f);
    }
}

static void sd_writer_task(void *pvParameters) {
    quad_buffer_t *qb = (quad_buffer_t *)pvParameters;
    
    ESP_LOGI(TAG, "SD writer task started");
    
    if (global_sd_enabled) {
        mkdir("/sdcard/data", 0755);
        load_state();
    }
    
    // allocate record buffer on heap
    sd_record_t *rec = malloc(sizeof(sd_record_t));
    if (!rec) {
        ESP_LOGE(TAG, "failed to allocate record buffer. task stopping.");
        vTaskDelete(NULL);
        return;
    }
    
    while (1) {
        int slot_idx;
        if (xQueueReceive(pending_sd_queue, &slot_idx, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        
        buffer_slot_t *slot = &qb->slots[slot_idx];
        
        if (global_sd_enabled && !slot->is_replay) {
            memset(rec, 0, sizeof(sd_record_t));
            rec->data.segment_id = slot->segment_id;
            rec->data.sample_rate = slot->sample_rate;
            
            memcpy(rec->data.accel_x, slot->data.accel_x, sizeof(float) * 4096);
            memcpy(rec->data.accel_y, slot->data.accel_y, sizeof(float) * 4096);
            memcpy(rec->data.accel_z, slot->data.accel_z, sizeof(float) * 4096);
            memcpy(rec->data.gyro_x, slot->data.gyro_x, sizeof(float) * 4096);
            memcpy(rec->data.gyro_y, slot->data.gyro_y, sizeof(float) * 4096);
            memcpy(rec->data.gyro_z, slot->data.gyro_z, sizeof(float) * 4096);
            
            rec->data.has_inference = slot->is_inferenced ? 1 : 0;
            if (slot->is_inferenced) {
                rec->data.router_model_id = slot->router_model_id;
                rec->data.memory_model_id = slot->memory_model_id;
                rec->data.active_submodel_id = slot->active_submodel_id;
                rec->data.anomaly_score = slot->anomaly_score;
                rec->data.is_anomaly = slot->is_anomaly ? 1 : 0;
            }
            
            // evaluate if this segment gets published in real-time
            bool sent_rt = false;
            extern uint32_t global_stream_mode;
            extern uint32_t global_cadence;
            if (global_stream_mode == 1) { // STREAM_MODE_CONTINUOUS
                sent_rt = true;
            } else if (global_stream_mode == 2) { // STREAM_MODE_ANOMALYONLY
                if (slot->is_anomaly) sent_rt = true;
            } else if (global_stream_mode == 3) { // STREAM_MODE_MIXED
                if (slot->is_anomaly) {
                    sent_rt = true;
                } else if (global_cadence > 0 && (slot->segment_id % global_cadence == 0)) {
                    sent_rt = true;
                }
            }
            rec->data.was_sent = sent_rt ? 1 : 0;
            
            char filepath[64];
            snprintf(filepath, sizeof(filepath), "/sdcard/data/data_%u.bin", (unsigned int)state.active_file_idx);
            
            FILE *f = fopen(filepath, state.record_count == 0 ? "wb" : "ab");
            if (f) {
                size_t written = fwrite(rec, 1, RECORD_SIZE, f);
                fclose(f);
                
                if (written == RECORD_SIZE) {
                    state.record_count++;
                    ESP_LOGI(TAG, "saved segment %u to %s (record %u/%d)", 
                             (unsigned int)slot->segment_id, filepath, 
                             (unsigned int)state.record_count, MAX_RECORDS_PER_FILE);
                    
                    if (state.record_count >= MAX_RECORDS_PER_FILE) {
                        state.record_count = 0;
                        state.active_file_idx = (state.active_file_idx + 1) % MAX_FILES;
                        ESP_LOGI(TAG, "file %s full, rotating to index %u", filepath, (unsigned int)state.active_file_idx);
                    }
                    
                    save_state();
                } else {
                    ESP_LOGE(TAG, "incomplete write to SD: %zu/%d bytes", written, RECORD_SIZE);
                    global_sd_enabled = false;
                }
            } else {
                ESP_LOGE(TAG, "failed to open %s", filepath);
                global_sd_enabled = false;
            }
        }
        
        quad_buffer_lock(slot);
        slot->flags &= ~BUF_FLAG_PENDING_SD;
        bool is_free = (slot->flags & (BUF_FLAG_READY_INF | BUF_FLAG_READY_SEND | BUF_FLAG_PENDING_SD)) == 0;
        quad_buffer_unlock(slot);
        
        if (is_free) {
            xQueueSend(free_slots_queue, &slot_idx, 0);
            ESP_LOGI(TAG, "slot %d released back to free", slot_idx);
        }
    }
    
    free(rec);
}

void sd_writer_start(quad_buffer_t *qb) {
    xTaskCreatePinnedToCore(
        sd_writer_task,
        "sd_writer",
        8192,
        qb,
        2,
        &sd_writer_task_handle,
        0
    );
}
