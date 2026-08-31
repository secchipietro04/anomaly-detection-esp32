#include "harvester.h"
#include "drivers/mems/ism330bx.h"
#include "esp_log.h"
#include "freertos/task.h"
#include "dsps_math.h"
#include "dsp_opt.h"
#include <string.h>

static const char *TAG = "harvester";

// sensor dev instance
static ism330bx_spi_dev_t dev;

// globals for config
extern float global_sample_rate;
extern bool global_sd_enabled;
extern bool global_inference_enabled;
extern TaskHandle_t inferencer_task_handle;
extern TaskHandle_t sd_writer_task_handle;
static uint32_t segment_id_counter = 1;
static ism330bx_config_t current_sensor_cfg;

void harvester_update_rate(float rate_hz) {
    global_sample_rate = rate_hz;
    current_sensor_cfg.accel_odr = ISM330BX_ODR_3840Hz;
    current_sensor_cfg.gyro_odr = ISM330BX_ODR_3840Hz;
    current_sensor_cfg.fifo_bdr_xl = ism330bx_hz_to_odr(rate_hz);
    current_sensor_cfg.fifo_bdr_gy = ism330bx_hz_to_odr(rate_hz);
    
    esp_err_t err = dev.apply_config(&dev, &current_sensor_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "failed to apply new ODR config: %d", err);
    } else {
        ESP_LOGI(TAG, "applied new sensor BDR matching rate %f Hz (ODR maxed out)", rate_hz);
    }
}
 
static void harvester_task(void *pvParameters) {
    quad_buffer_t *qb = (quad_buffer_t *)pvParameters;
    
    ism330bx_init_config_t init_cfg = ISM330BX_DEFAULT_INIT_CONFIG();
    init_cfg.sensor.accel_odr = ISM330BX_ODR_3840Hz;
    init_cfg.sensor.gyro_odr = ISM330BX_ODR_3840Hz;
    init_cfg.sensor.fifo_bdr_xl = ism330bx_hz_to_odr(global_sample_rate);
    init_cfg.sensor.fifo_bdr_gy = ism330bx_hz_to_odr(global_sample_rate);
    current_sensor_cfg = init_cfg.sensor;
    
    ESP_LOGI(TAG, "Initializing SPI sensor...");
    esp_err_t err = ism330bx_spi_create(&dev, &init_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "sensor create failed: %d. harvester stopping.", err);
        vTaskDelete(NULL);
        return;
    }
    
    err = dev.init(&dev);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "sensor init failed: %d. harvester stopping.", err);
        vTaskDelete(NULL);
        return;
    }
    
    ESP_LOGI(TAG, "sensor up. starting harvester loop");
    
    int16_t *ax_raw = malloc(sizeof(int16_t) * 512);
    int16_t *ay_raw = malloc(sizeof(int16_t) * 512);
    int16_t *az_raw = malloc(sizeof(int16_t) * 512);
    int16_t *gx_raw = malloc(sizeof(int16_t) * 512);
    int16_t *gy_raw = malloc(sizeof(int16_t) * 512);
    int16_t *gz_raw = malloc(sizeof(int16_t) * 512);
    if (!ax_raw || !ay_raw || !az_raw || !gx_raw || !gy_raw || !gz_raw) {
        ESP_LOGE(TAG, "out of memory for temp raw arrays");
        free(ax_raw); free(ay_raw); free(az_raw);
        free(gx_raw); free(gy_raw); free(gz_raw);
        vTaskDelete(NULL);
        return;
    }
    
    uint32_t loop_counter = 0;
    
    while (1) {
        int slot_idx;
        if (xQueueReceive(free_slots_queue, &slot_idx, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        buffer_slot_t *slot = &qb->slots[slot_idx];
        
        quad_buffer_lock(slot);
        slot->flags = BUF_FLAG_HARVESTING;
        quad_buffer_unlock(slot);
        
        uint32_t accel_idx = 0;
        uint32_t gyro_idx = 0;
        bool skip_id = false;
        
        while (accel_idx < SAMPLES_PER_SEGMENT || gyro_idx < SAMPLES_PER_SEGMENT) {
            ism330bx_fifo_result_t result = {
                .accel_x = ax_raw,
                .accel_y = ay_raw,
                .accel_z = az_raw,
                .accel_data_size = 512,
                .gyro_x = gx_raw,
                .gyro_y = gy_raw,
                .gyro_z = gz_raw,
                .gyro_data_size = 512
            };
            
            err = dev.fetch_fifo_buffer(&dev, &result);
            if (err != ESP_OK) {
                taskYIELD();
                continue;
            }
            
            uint8_t stat2 = 0;
            dev.read_reg(&dev, ISM330BX_REG_FIFO_STATUS2, &stat2, 1);
            if (stat2 & ISM330BX_FIFO_STAT2_OVR) {
                skip_id = true;
            }
            
            if (result.accel_count > 0 && accel_idx < SAMPLES_PER_SEGMENT) {
                size_t to_copy = (accel_idx + result.accel_count > SAMPLES_PER_SEGMENT) ? (SAMPLES_PER_SEGMENT - accel_idx) : result.accel_count;
                dsp_opt_int16_to_float(ax_raw, &slot->data.accel_x[accel_idx], to_copy);
                dsp_opt_int16_to_float(ay_raw, &slot->data.accel_y[accel_idx], to_copy);
                dsp_opt_int16_to_float(az_raw, &slot->data.accel_z[accel_idx], to_copy);
                accel_idx += to_copy;
            }
            
            if (result.gyro_count > 0 && gyro_idx < SAMPLES_PER_SEGMENT) {
                size_t to_copy = (gyro_idx + result.gyro_count > SAMPLES_PER_SEGMENT) ? (SAMPLES_PER_SEGMENT - gyro_idx) : result.gyro_count;
                dsp_opt_int16_to_float(gx_raw, &slot->data.gyro_x[gyro_idx], to_copy);
                dsp_opt_int16_to_float(gy_raw, &slot->data.gyro_y[gyro_idx], to_copy);
                dsp_opt_int16_to_float(gz_raw, &slot->data.gyro_z[gyro_idx], to_copy);
                gyro_idx += to_copy;
            }
            
            if (result.accel_count == 0 && result.gyro_count == 0) {
                taskYIELD();
            }
        }
        
        slot->sample_rate = global_sample_rate;
        slot->segment_id = segment_id_counter;
        if (skip_id) {
            ESP_LOGW(TAG, "sensor FIFO overrun detected, skipping an ID");
            segment_id_counter += 2;
        } else {
            segment_id_counter += 1;
        }
        
        ESP_LOGI(TAG, "slot %d harvested 4096 samples (segment ID %u)", slot_idx, (unsigned int)slot->segment_id);
        
        quad_buffer_lock(slot);
        slot->flags = 0;
        
        bool run_inf = global_inference_enabled;
        bool write_sd = global_sd_enabled;
        
        if (run_inf) {
            slot->flags |= BUF_FLAG_READY_INF;
        } else {
            slot->flags |= BUF_FLAG_READY_SEND;
        }
        
        if (write_sd) {
            slot->flags |= BUF_FLAG_PENDING_SD;
        }
        quad_buffer_unlock(slot);
        
        if (run_inf) {
            xQueueSend(ready_inf_queue, &slot_idx, 0);
        } else {
            xQueueSend(ready_send_queue, &slot_idx, 0);
        }
        if (write_sd) {
            xQueueSend(pending_sd_queue, &slot_idx, 0);
        }
        
        qb->write_idx = (qb->write_idx + 1) % 4;
    }
}

void harvester_start(quad_buffer_t *qb) {
    xTaskCreatePinnedToCore(
        harvester_task,
        "harvester",
        16384,
        qb,
        5,
        NULL,
        1
    );
}
