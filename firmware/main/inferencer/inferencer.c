#include "inferencer.h"
#include "model_ensemble.h"
#include "dsp_utils.h"
#include "cbor/gen/ensemble_types.h"
#include "cbor/gen/ensemble.h"
#include "esp_fft_wrapper.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "drivers/sd_card/sd_card.h"
#include "network/mqtt/mqtt.h"
#include <string.h>
#include <math.h>

static const char *TAG = "inferencer";

// TF Lite model ensemble singleton instance
model_ensemble_t ensemble;
static bool ensemble_initialized = false;

// Task handle
TaskHandle_t inferencer_task_handle = NULL;

// Globals
extern char global_node_id[32];
extern mqtt_wrapper_t *global_mqtt_client;
extern bool global_sd_enabled;
extern uint32_t global_last_segment_id;

// Semaphore to block inferencer task waiting for dynamic submodel download
static SemaphoreHandle_t mqtt_download_sem = NULL;
static const uint8_t *downloaded_cbor_payload = NULL;
static size_t downloaded_cbor_size = 0;

static void dynamic_model_cb(const uint8_t *data, size_t len, void *user_ctx) {
    ESP_LOGI(TAG, "received dynamic model binary payload (%zu bytes)", len);
    downloaded_cbor_payload = data;
    downloaded_cbor_size = len;
    if (mqtt_download_sem) {
        xSemaphoreGive(mqtt_download_sem);
    }
}

// dynamic fetching from storage with cloud MQTT fallback
static const uint8_t* fetch_model_cbor(uint32_t model_id, size_t *out_len, bool *out_must_free) {
    *out_must_free = false;
    
    // Step 1: Check SD card storage first if enabled
    if (global_sd_enabled) {
        char path[64];
        snprintf(path, sizeof(path), "/sdcard/models/model_%u.bin", (unsigned int)model_id);
        size_t file_len = 0;
        uint8_t *data = sd_card_read_file(path, &file_len);
        if (data && file_len > 0) {
            ESP_LOGI(TAG, "hit SD card storage cache for model %u (%zu bytes)", (unsigned int)model_id, file_len);
            *out_len = file_len;
            *out_must_free = true;
            return data;
        }
    }
    
    // Step 2: Query Cloud Backend via MQTT Request-Response
    if (!global_mqtt_client || !global_mqtt_client->is_connected(global_mqtt_client)) {
        ESP_LOGE(TAG, "cannot fetch submodel %u: MQTT not connected", (unsigned int)model_id);
        return NULL;
    }
    
    char sub_topic[128];
    snprintf(sub_topic, sizeof(sub_topic), "v1/%s/models/submodel/%u", global_node_id, (unsigned int)model_id);
    
    downloaded_cbor_payload = NULL;
    downloaded_cbor_size = 0;
    
    // subscribe to dedicated single-model endpoint
    global_mqtt_client->subscribe(global_mqtt_client, sub_topic, 1, dynamic_model_cb, NULL);
    
    // trigger on-demand fetch
    char fetch_topic[128];
    snprintf(fetch_topic, sizeof(fetch_topic), "v1/%s/models/fetch/submodel/%u", global_node_id, (unsigned int)model_id);
    
    uint8_t dummy = 0;
    global_mqtt_client->publish(global_mqtt_client, fetch_topic, &dummy, 0, 1, 0);
    ESP_LOGI(TAG, "published fetch trigger request for model %u", (unsigned int)model_id);
    
    // wait with timeout (5000 ms) for MQTT download response
    if (xSemaphoreTake(mqtt_download_sem, pdMS_TO_TICKS(5000)) == pdTRUE && downloaded_cbor_payload != NULL) {
        ESP_LOGI(TAG, "successfully downloaded submodel %u from backend", (unsigned int)model_id);
        
        // save to SD storage cache
        if (global_sd_enabled) {
            char path[64];
            snprintf(path, sizeof(path), "/sdcard/models/model_%u.bin", (unsigned int)model_id);
            sd_card_save_file(path, downloaded_cbor_payload, downloaded_cbor_size);
        }
        
        *out_len = downloaded_cbor_size;
        *out_must_free = false; // payload memory belongs to MQTT receive buffer
        
        // unsubscribe after receiving
        global_mqtt_client->unsubscribe(global_mqtt_client, sub_topic);
        return downloaded_cbor_payload;
    }
    
    ESP_LOGE(TAG, "timed out waiting for submodel %u from backend", (unsigned int)model_id);
    global_mqtt_client->unsubscribe(global_mqtt_client, sub_topic);
    return NULL;
}

static bool load_model_package(const uint8_t *cbor_data, size_t cbor_len, uint32_t model_id, bool free_cbor) {
    if (!cbor_data) return false;
    
    // decode the cbor package
    struct Payload_r payload;
    size_t decoded_len = 0;
    int dec_res = cbor_decode_Payload(cbor_data, cbor_len, &payload, &decoded_len);
    bool success = (dec_res == 0);
    if (!success || payload.Payload_choice != Payload_ModelPackage_m_c) {
        ESP_LOGE(TAG, "cbor decoding of model package failed (res: %d)", dec_res);
        if (free_cbor) free((void*)cbor_data);
        return false;
    }
    
    struct ModelPackage_r *pkg = &payload.Payload_ModelPackage_m;
    if (pkg->ModelPackage_choice != AutoencoderModelPackage_m_c) {
        ESP_LOGE(TAG, "expected autoencoder submodel package choice");
        if (free_cbor) free((void*)cbor_data);
        return false;
    }
    
    struct AutoencoderModelPackage *ae_pkg = &pkg->AutoencoderModelPackage_m;
    struct BaseModelPackage_r *base = &ae_pkg->AutoencoderModelPackage_BaseModelPackage_m;
    
    // Load submodel flatbuffer directly into ensemble cache
    int load_res = model_ensemble_load_submodel(&ensemble, base->BaseModelPackage_data.value, base->BaseModelPackage_data.len);
    if (load_res != MODEL_SUCCESS) {
        ESP_LOGE(TAG, "model_ensemble_load_submodel failed (res: %d)", load_res);
        if (free_cbor) free((void*)cbor_data);
        return false;
    }
    
    if (free_cbor) free((void*)cbor_data);
    ESP_LOGI(TAG, "successfully loaded and cached submodel %u into memory arena", (unsigned int)model_id);
    return true;
}

static bool fetch_and_load_submodel(uint32_t model_id) {
    // 1. query cache pool
    ModelInstance_t* cached = model_cache_get(&ensemble.submodel_cache, model_id);
    if (cached != NULL) {
        return true; // Already loaded and hot in arena
    }
    
    // 2. fetch from local storage or cloud
    size_t cbor_len = 0;
    bool free_cbor = false;
    const uint8_t *cbor_data = fetch_model_cbor(model_id, &cbor_len, &free_cbor);
    
    if (!cbor_data) {
        ESP_LOGE(TAG, "failed to get model %u bytes", (unsigned int)model_id);
        return false;
    }
    
    // 3. decode and load into the TF Lite ensemble cache
    return load_model_package(cbor_data, cbor_len, model_id, free_cbor);
}

// compute 1D signal magnitude and 2D STFT spectrograms from raw 3-axis readings
static void compute_signal_features(const buffer_slot_t *slot, float *accel_mag, float *gyro_mag, float *accel_spec, float *gyro_spec) {
    // step 1: magnitude conversion
    for (int n = 0; n < SAMPLES_PER_SEGMENT; n++) {
        float ax = slot->data.accel_x[n];
        float ay = slot->data.accel_y[n];
        float az = slot->data.accel_z[n];
        accel_mag[n] = sqrtf(ax*ax + ay*ay + az*az);
        
        float gx = slot->data.gyro_x[n];
        float gy = slot->data.gyro_y[n];
        float gz = slot->data.gyro_z[n];
        gyro_mag[n] = sqrtf(gx*gx + gy*gy + gz*gz);
    }
    
    // step 2: windowing & STFT
    esp_fft_wrapper_stft(accel_mag, SAMPLES_PER_SEGMENT, FFT_WINDOW_SIZE, FFT_HOP_SIZE, FFT_SIZE, accel_spec);
    esp_fft_wrapper_stft(gyro_mag, SAMPLES_PER_SEGMENT, FFT_WINDOW_SIZE, FFT_HOP_SIZE, FFT_SIZE, gyro_spec);
}

// loop over spectrogram frames, run the ensemble routing, and run submodel inferences
static void run_ensemble_inference(buffer_slot_t *slot, const float *accel_spec, const float *gyro_spec) {
    float accel_pooled[128];
    float gyro_pooled[128];
    float concatenated_slice[DEFAULT_RAW_BINS];
    
    uint32_t active_router_id = ensemble.router_model_loaded ? ensemble.router_model.config.model_id : 0;
    uint32_t active_memory_id = ensemble.memory_model_loaded ? ensemble.memory_model.config.model_id : 0;
    
    // 1. Run memory backbone step-by-step for all frames to propagate recurrent state
    if (ensemble.memory_model_loaded) {
        for (int t = 0; t < SPECTROGRAM_FRAMES; t++) {
            pool_1d_max_pow2(&accel_spec[t * SPECTROGRAM_BINS], 128, accel_pooled, 128);
            pool_1d_max_pow2(&gyro_spec[t * SPECTROGRAM_BINS], 128, gyro_pooled, 128);
            memcpy(concatenated_slice, accel_pooled, 128 * sizeof(float));
            memcpy(concatenated_slice + 128, gyro_pooled, 128 * sizeof(float));
            model_ensemble_inf_memory(&ensemble, concatenated_slice);
        }
    }
    
    // 2. Evaluate the router ONCE for the entire segment
    uint32_t target_submodel = 0;
    bool already_loaded = false;
    bool router_anomaly = false;
    
    if (ensemble.router_model_loaded) {
        bool router_ok = model_ensemble_inf_router(&ensemble, &target_submodel, &already_loaded, &router_anomaly);
        if (!router_ok) {
            ESP_LOGE(TAG, "Router model invocation failed");
            return;
        }
        
        float max_anomaly_score = 0.0f;
        bool is_any_anomaly = false;
        uint32_t resolved_submodel_id = 0;
        
        if (router_anomaly) {
            max_anomaly_score = 1.0f;
            is_any_anomaly = true;
            resolved_submodel_id = 0;
            ESP_LOGW(TAG, "Ensemble router failed to match any submodel. Classified as Router Anomaly!");
        } else {
            resolved_submodel_id = target_submodel;
            
            if (!already_loaded) {
                if (!fetch_and_load_submodel(target_submodel)) {
                    ESP_LOGE(TAG, "Failed to load submodel %u from storage/cloud", (unsigned int)target_submodel);
                    return;
                }
            }
            
            uint32_t skip_amt = ensemble.submodel_cache.items[0].model.config.skip_amount;
            float score = 0.0f;
            bool is_anom = false;
            
            bool ae_ok = model_ensemble_inf_ae(&ensemble, &score, &is_anom, SPECTROGRAM_FRAMES, skip_amt);
            if (ae_ok) {
                max_anomaly_score = score;
                is_any_anomaly = is_anom;
            }
        }
        
        slot->router_model_id = active_router_id;
        slot->memory_model_id = active_memory_id;
        slot->active_submodel_id = resolved_submodel_id;
        slot->anomaly_score = max_anomaly_score;
        slot->is_anomaly = is_any_anomaly;
        slot->is_inferenced = true;
        global_last_segment_id = slot->segment_id;
        
        ESP_LOGI(TAG, "inference completed for segment %u. Max score: %f, Anomaly: %d", 
                 (unsigned int)slot->segment_id, max_anomaly_score, is_any_anomaly);
    } else {
        slot->is_inferenced = false;
        global_last_segment_id = slot->segment_id;
        ESP_LOGI(TAG, "warmup data collection phase (no models loaded yet). segment %u ready for telemetry uplink", (unsigned int)slot->segment_id);
    }
}

static void inferencer_task(void *pvParameters) {
    quad_buffer_t *qb = (quad_buffer_t *)pvParameters;
    
    // allocate semaphore for download sync
    mqtt_download_sem = xSemaphoreCreateBinary();
    
    // setup default ensemble config
    if (!model_ensemble_init(&ensemble, DEFAULT_RAW_BINS, DEFAULT_HISTORY_DEPTH, DEFAULT_WARMUP_STEPS)) {
        ESP_LOGE(TAG, "failed to init model ensemble wrapper");
        vTaskDelete(NULL);
        return;
    }
    ensemble_initialized = true;
    
    // init fft table
    esp_fft_wrapper_init(FFT_SIZE);
    
    ESP_LOGI(TAG, "inferencer thread started");
    
    float *accel_mag = malloc(sizeof(float) * SAMPLES_PER_SEGMENT);
    float *gyro_mag = malloc(sizeof(float) * SAMPLES_PER_SEGMENT);

    // spectrograms ((total samples - window size)/Hop) + 1
    float *accel_spec = malloc(sizeof(float) * SPECTROGRAM_FRAMES * SPECTROGRAM_BINS);
    float *gyro_spec = malloc(sizeof(float) * SPECTROGRAM_FRAMES * SPECTROGRAM_BINS);
    
    if (!accel_mag || !gyro_mag || !accel_spec || !gyro_spec) {
        ESP_LOGE(TAG, "out of memory for dsp temp buffers");
        vTaskDelete(NULL);
        return;
    }
    
    while (1) {
        // wait for harvester notification via queue
        int slot_idx;
        if (xQueueReceive(ready_inf_queue, &slot_idx, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        
        buffer_slot_t *slot = &qb->slots[slot_idx];
        
        // run only on ready inf slots
        if (slot->flags & BUF_FLAG_READY_INF) {
            ESP_LOGI(TAG, "processing slot %d (segment ID %u)", slot_idx, (unsigned int)slot->segment_id);
            
#if defined(CONFIG_RECORD_INFERENCE_TIME) || defined(RECORD_INFERENCE_TIME)
            int64_t inf_start = esp_timer_get_time();
#endif
            compute_signal_features(slot, accel_mag, gyro_mag, accel_spec, gyro_spec);
            run_ensemble_inference(slot, accel_spec, gyro_spec);
#if defined(CONFIG_RECORD_INFERENCE_TIME) || defined(RECORD_INFERENCE_TIME)
            slot->inference_time_ms = (uint32_t)((esp_timer_get_time() - inf_start) / 1000);
#endif
            
            // lock slot and change flags
            quad_buffer_lock(slot);
            slot->flags &= ~BUF_FLAG_READY_INF; // clear inferencing flag
            slot->flags &= ~BUF_FLAG_FORCE_INF; // clear force inf flag
            slot->flags |= BUF_FLAG_READY_SEND;  // flag uploader
            quad_buffer_unlock(slot);
            
            // notify uploader thread via queue
            xQueueSend(ready_send_queue, &slot_idx, 0);
        }
    }
}

void inferencer_start(quad_buffer_t *qb) {
    xTaskCreatePinnedToCore(
        inferencer_task,
        "inferencer",
        16384,
        qb,
        4,
        &inferencer_task_handle,
        1
    );
}
