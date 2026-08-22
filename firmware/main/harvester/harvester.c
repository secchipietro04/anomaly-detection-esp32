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
    // change the rate in runtime config
    global_sample_rate = rate_hz;
    
    current_sensor_cfg.accel_odr = ISM330BX_ODR_3840Hz; // Always max out physical ODR
    current_sensor_cfg.gyro_odr = ISM330BX_ODR_3840Hz;  // Always max out physical ODR
    current_sensor_cfg.fifo_bdr_xl = ism330bx_hz_to_odr(rate_hz);
    current_sensor_cfg.fifo_bdr_gy = ism330bx_hz_to_odr(rate_hz);
    
    // apply to driver
    esp_err_t err = dev.apply_config(&dev, &current_sensor_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "failed to apply new ODR config: %d", err);
    } else {
        ESP_LOGI(TAG, "applied new sensor BDR matching rate %f Hz (ODR maxed out)", rate_hz);
    }
}
 
static void harvester_task(void *pvParameters) {
    quad_buffer_t *qb = (quad_buffer_t *)pvParameters;
    
    // setup default init configuration
    ism330bx_init_config_t init_cfg = ISM330BX_DEFAULT_INIT_CONFIG();
    current_sensor_cfg = init_cfg.sensor;
    
    // force initial rate from globals
    current_sensor_cfg.accel_odr = ISM330BX_ODR_3840Hz;
    current_sensor_cfg.gyro_odr = ISM330BX_ODR_3840Hz;
    current_sensor_cfg.fifo_bdr_xl = ism330bx_hz_to_odr(global_sample_rate);
    current_sensor_cfg.fifo_bdr_gy = ism330bx_hz_to_odr(global_sample_rate);
    
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
    
    // allocate buffers on heap for reading the fifo tag packets
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
    
    uint32_t sample_idx = 0;
    
    while (1) {
        // retrieve free slot index from event queue (blocks until a slot becomes free)
        int slot_idx;
        if (xQueueReceive(free_slots_queue, &slot_idx, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        buffer_slot_t *slot = &qb->slots[slot_idx];
        
        // lock and set harvesting flag
        quad_buffer_lock(slot);
        slot->flags = BUF_FLAG_HARVESTING;
        quad_buffer_unlock(slot);
        
        sample_idx = 0;
        bool skip_id = false;
        
        // fill current slot to 4096 samples
        while (sample_idx < 4096) {
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
            
            // fetch what is available in fifo
            err = dev.fetch_fifo_buffer(&dev, &result);
            if (err != ESP_OK) {
                vTaskDelay(pdMS_TO_TICKS(2));
                continue;
            }
            
            // check overrun status
            uint8_t stat2 = 0;
            dev.read_reg(&dev, ISM330BX_REG_FIFO_STATUS2, &stat2, 1);
            if (stat2 & ISM330BX_FIFO_STAT2_OVR) {
                // we had an overrun, skip segment ID
                skip_id = true;
            }
            
            size_t copy_cnt = (result.accel_count < result.gyro_count) ? result.accel_count : result.gyro_count;
            if (sample_idx + copy_cnt > SAMPLES_PER_SEGMENT) {
                copy_cnt = SAMPLES_PER_SEGMENT - sample_idx;
            }
            if (copy_cnt > 0) {
                dsp_opt_int16_to_float(ax_raw, &slot->data.accel_x[sample_idx], copy_cnt);
                dsp_opt_int16_to_float(ay_raw, &slot->data.accel_y[sample_idx], copy_cnt);
                dsp_opt_int16_to_float(az_raw, &slot->data.accel_z[sample_idx], copy_cnt);
                dsp_opt_int16_to_float(gx_raw, &slot->data.gyro_x[sample_idx], copy_cnt);
                dsp_opt_int16_to_float(gy_raw, &slot->data.gyro_y[sample_idx], copy_cnt);
                dsp_opt_int16_to_float(gz_raw, &slot->data.gyro_z[sample_idx], copy_cnt);
                sample_idx += copy_cnt;
            }
            
            vTaskDelay(pdMS_TO_TICKS(10));
        }
        
        // filled buffer slot
        slot->sample_rate = global_sample_rate;
        slot->segment_id = segment_id_counter;
        if (skip_id) {
            ESP_LOGW(TAG, "sensor FIFO overrun detected, skipping an ID");
            segment_id_counter += 2; // skip one id
        } else {
            segment_id_counter += 1;
        }
        
        // lock and set ready flags
        quad_buffer_lock(slot);
        slot->flags = 0; // clear harvesting flag
        
        bool run_inf = global_inference_enabled;
        bool write_sd = global_sd_enabled;
        
        if (run_inf) {
            slot->flags |= BUF_FLAG_READY_INF;
        } else {
            slot->flags |= BUF_FLAG_READY_SEND; // bypass directly to uploader
        }
        
        if (write_sd) {
            slot->flags |= BUF_FLAG_PENDING_SD;
        }
        quad_buffer_unlock(slot);
        
        // Notify other threads via the event queues
        if (run_inf) {
            xQueueSend(ready_inf_queue, &slot_idx, 0);
        } else {
            xQueueSend(ready_send_queue, &slot_idx, 0);
        }
        if (write_sd) {
            xQueueSend(pending_sd_queue, &slot_idx, 0);
        }
        
        // rotate write index (compatibility)
        qb->write_idx = (qb->write_idx + 1) % 4;
    }
}

void harvester_start(quad_buffer_t *qb) {
    // start core 1 task for sensor harvesting
    xTaskCreatePinnedToCore(
        harvester_task,
        "harvester",
        8192,
        qb,
        5,
        NULL,
        1
    );
}
