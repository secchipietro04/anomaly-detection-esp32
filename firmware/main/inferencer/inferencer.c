#include "inferencer.h"
#include "esp_fft_wrapper.h"
#include "model_ensemble.h"
#include "dsp_utils.h"
#include "loss_utils.h"
#include "cbor/gen/ensemble.h"
#include "network/mqtt/mqtt.h"
#include "drivers/sd_card/sd_card.h"
#include "esp_log.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <string.h>
#include <math.h>
#include <sys/unistd.h>
#include <sys/stat.h>

static const char *TAG = "inferencer";

// globals
extern float global_sample_rate;
extern bool global_sd_enabled;
extern TaskHandle_t uploader_task_handle;
extern mqtt_wrapper_t *global_mqtt_client;
extern char global_node_id[32];
extern uint32_t global_last_segment_id;

// active ensemble instance
model_ensemble_t ensemble;
static bool ensemble_initialized = false;

// sync semaphore for mqtt model download
static SemaphoreHandle_t mqtt_download_sem = NULL;
static uint8_t *downloaded_cbor = NULL;
static size_t downloaded_cbor_len = 0;
static uint32_t current_downloading_model_id = 0;

TaskHandle_t inferencer_task_handle = NULL;



// mqtt callback for receiving model packages
static void model_mqtt_callback(const uint8_t *data, size_t len, void *user_ctx) {
    // allocate buffer and copy cbor payload
    if (downloaded_cbor) {
        free(downloaded_cbor);
        downloaded_cbor = NULL;
    }
    
    downloaded_cbor = malloc(len);
    if (downloaded_cbor) {
        memcpy(downloaded_cbor, data, len);
        downloaded_cbor_len = len;
    }
    
    // wake up the waiting task
    if (mqtt_download_sem) {
        xSemaphoreGive(mqtt_download_sem);
    }
}

// check if model is saved on SD card
static uint8_t* load_model_from_sd(uint32_t model_id, size_t* out_size) {
    *out_size = 0;
    if (!global_sd_enabled) {
        return NULL;
    }
    char path[64];
    snprintf(path, sizeof(path), "/sdcard/models/model_%u.bin", (unsigned int)model_id);
    uint8_t *cbor_data = sd_card_read_file(path, out_size);
    if (cbor_data) {
        ESP_LOGI(TAG, "loaded model %u from SD card", (unsigned int)model_id);
    }
    return cbor_data;
}

// fetch model from cloud via MQTT
static uint8_t* download_model(uint32_t model_id, size_t* out_size) {
    *out_size = 0;
    if (!global_mqtt_client || !global_mqtt_client->is_connected(global_mqtt_client)) {
        return NULL;
    }
    
    char path[64];
    snprintf(path, sizeof(path), "/sdcard/models/model_%u.bin", (unsigned int)model_id);
    char topic[128];
    snprintf(topic, sizeof(topic), "v1/%s/model/%u", global_node_id, (unsigned int)model_id);
    
    ESP_LOGI(TAG, "model %u not on SD. subscribing to topic %s", (unsigned int)model_id, topic);
    current_downloading_model_id = model_id;
    
    // reset download pointers
    if (downloaded_cbor) {
        free(downloaded_cbor);
        downloaded_cbor = NULL;
    }
    downloaded_cbor_len = 0;
    
    xSemaphoreTake(mqtt_download_sem, 0); // clear sem
    
    uint8_t *cbor_data = NULL;
    int sub_res = global_mqtt_client->subscribe(global_mqtt_client, topic, 1, model_mqtt_callback, NULL);
    if (sub_res >= 0) {
        // wait for data (15 sec timeout)
        if (xSemaphoreTake(mqtt_download_sem, pdMS_TO_TICKS(15000)) == pdTRUE) {
            cbor_data = downloaded_cbor;
            *out_size = downloaded_cbor_len;
            
            // save downloaded model to SD
            if (global_sd_enabled && cbor_data) {
                sd_card_save_file(path, cbor_data, *out_size);
            }
        } else {
            ESP_LOGE(TAG, "timeout downloading model %u", (unsigned int)model_id);
        }
        global_mqtt_client->unsubscribe(global_mqtt_client, topic);
    }
    
    return cbor_data;
}

// decode and load the model package into the TF Lite ensemble cache
static bool load_model_package(const uint8_t* cbor_data, size_t cbor_len, uint32_t model_id, bool free_cbor) {
    if (!cbor_data) return false;
    
    // decode the cbor package
    struct Payload_r payload;
    size_t decoded_len = 0;
    bool success = cbor_decode_Payload(cbor_data, cbor_len, &payload, &decoded_len);
    
    if (!success || payload.Payload_choice != Payload_ModelPackage_m_c) {
        ESP_LOGE(TAG, "cbor decoding of model package failed");
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
    
    // loop until it fits in RAM (evict cached models if out of memory)
    int load_res = MODEL_ERR_GENERIC;
    uint32_t current_cap = ensemble.submodel_cache.capacity;
    
    while (current_cap > 0) {
        load_res = model_ensemble_load_submodel(&ensemble, base->BaseModelPackage_data.value, base->BaseModelPackage_data.len);
        if (load_res == MODEL_SUCCESS) {
            // loaded! now configure the loaded model metadata
            ModelInstance_t* mi = &ensemble.submodel_cache.items[0].model;
            mi->config.model_id = base->BaseModelPackage_m_id;
            mi->config.archetype = ARCHETYPE_AUTOENCODER;
            mi->config.temporal_depth = ae_pkg->AutoencoderModelPackage_tsteps;
            mi->config.frequency_bins = ae_pkg->AutoencoderModelPackage_accel_bins + ae_pkg->AutoencoderModelPackage_gyro_bins;
            mi->config.anomaly_threshold = ae_pkg->AutoencoderModelPackage_limit;
            
            // set loss mode
            mi->config.loss_mode = (ae_pkg->AutoencoderModelPackage_loss.LossMode_choice == LossMode_LogMSE_m_c) ? LOSS_MODE_LOG_MSE : LOSS_MODE_LINEAR_MSE;
            mi->config.skip_amount = ae_pkg->AutoencoderModelPackage_skip_present ? ae_pkg->AutoencoderModelPackage_skip.AutoencoderModelPackage_skip : 0;
            
            ESP_LOGI(TAG, "submodel %u successfully loaded into cache", (unsigned int)model_id);
            break;
        } else if (load_res == MODEL_ERR_NO_MEM) {
            // shrink cache limit to evict models and try again
            current_cap--;
            ESP_LOGW(TAG, "OOM loading submodel. shrinking cache capacity to %u", (unsigned int)current_cap);
            model_ensemble_set_cache_capacity(&ensemble, current_cap);
        } else {
            ESP_LOGE(TAG, "fatal error loading submodel flatbuffer: %d", load_res);
            break;
        }
    }
    
    if (free_cbor) free((void*)cbor_data);
    return (load_res == MODEL_SUCCESS);
}

// load model from sd or fetch via mqtt
static bool fetch_and_load_submodel(uint32_t model_id) {
    size_t cbor_len = 0;
    bool free_cbor = false;
    
    // 1. check SD card first
    uint8_t *cbor_data = load_model_from_sd(model_id, &cbor_len);
    if (cbor_data) {
        free_cbor = true; // we must free it
    } else {
        // 2. if not found, download from cloud
        cbor_data = download_model(model_id, &cbor_len);
        free_cbor = false; // download buffer is owned by the global downloaded_cbor pointer, do not free it
    }
    
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
    for (int t = 0; t < SPECTROGRAM_FRAMES; t++) {
        // Pool accel & gyro to 128 bins each (direct copy)
        pool_1d_max_pow2(&accel_spec[t * SPECTROGRAM_BINS], 128, accel_pooled, 128);
        pool_1d_max_pow2(&gyro_spec[t * SPECTROGRAM_BINS], 128, gyro_pooled, 128);
        
        // Concatenate to 256-float slice
        memcpy(concatenated_slice, accel_pooled, 128 * sizeof(float));
        memcpy(concatenated_slice + 128, gyro_pooled, 128 * sizeof(float));
        
        model_ensemble_inf_memory(&ensemble, concatenated_slice);
    }
    
    // 2. Evaluate the router ONCE for the entire segment
    uint32_t target_submodel = 0;
    bool already_loaded = false;
    bool router_anomaly = false;
    
    bool router_ok = model_ensemble_inf_router(&ensemble, &target_submodel, &already_loaded, &router_anomaly);
    if (!router_ok) {
        ESP_LOGE(TAG, "Router model invocation failed");
        return;
    }
    
    float max_anomaly_score = 0.0f;
    bool is_any_anomaly = false;
    uint32_t resolved_submodel_id = 0;
    
    if (router_anomaly) {
        // Router classification confidence was too low -> instantly flag as anomaly!
        max_anomaly_score = 1.0f; // maximum anomaly score
        is_any_anomaly = true;
        resolved_submodel_id = 0;
        ESP_LOGW(TAG, "Ensemble router failed to match any submodel. Classified as Router Anomaly!");
    } else {
        resolved_submodel_id = target_submodel;
        
        // If submodel is not already cached, fetch and load it
        if (!already_loaded) {
            if (!fetch_and_load_submodel(target_submodel)) {
                ESP_LOGE(TAG, "Failed to load submodel %u from storage/cloud", (unsigned int)target_submodel);
                return;
            }
        }
        
        // 3. Run the active Autoencoder submodel using its model-specific skip amount
        uint32_t skip_amt = ensemble.submodel_cache.items[0].model.config.skip_amount;
        float score = 0.0f;
        bool is_anom = false;
        
        bool ae_ok = model_ensemble_inf_ae(&ensemble, &score, &is_anom, SPECTROGRAM_FRAMES, skip_amt);
        if (ae_ok) {
            max_anomaly_score = score;
            is_any_anomaly = is_anom;
        }
    }
    
    // save inference results to slot
    slot->router_model_id = active_router_id;
    slot->memory_model_id = active_memory_id;
    slot->active_submodel_id = resolved_submodel_id;
    slot->anomaly_score = max_anomaly_score;
    slot->is_anomaly = is_any_anomaly;
    slot->is_inferenced = true;
    global_last_segment_id = slot->segment_id;
    
    ESP_LOGI(TAG, "inference completed for segment %u. Max score: %f, Anomaly: %d", 
             (unsigned int)slot->segment_id, max_anomaly_score, is_any_anomaly);
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

    //spectrograms     ((total samples - window size)/Hop) +1
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
            
            compute_signal_features(slot, accel_mag, gyro_mag, accel_spec, gyro_spec);
            run_ensemble_inference(slot, accel_spec, gyro_spec);
            
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
