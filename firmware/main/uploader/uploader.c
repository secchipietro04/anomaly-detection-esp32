#include "uploader.h"
#include "cbor/gen/telemetry_encode.h"
#include "cbor/gen/device_types.h"
#include "network/mqtt/mqtt.h"
#include "esp_log.h"
#include "freertos/task.h"
#include "dump_reader.h"
#include <stdlib.h>
#include <string.h>

static const char *TAG = "uploader";

TaskHandle_t uploader_task_handle = NULL;

// configs
extern mqtt_wrapper_t *global_mqtt_client;
extern char global_node_id[32];
extern uint32_t global_stream_mode;
extern uint32_t global_cadence;

static bool serialize_and_publish_segment(buffer_slot_t *slot, int32_t reason) {
    size_t max_cbor_len = 16 * 1024;
    uint8_t *buf = malloc(max_cbor_len);
    if (!buf) {
        ESP_LOGE(TAG, "failed to alloc cbor segment buffer");
        return false;
    }
    
    struct Segment seg;
    bool overall_ok = true;
    
    uint32_t num_chunks = SAMPLES_PER_SEGMENT / TELEMETRY_CHUNK_SIZE;
    
    for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        memset(&seg, 0, sizeof(struct Segment));
        
        seg.Segment_id = slot->segment_id;
        seg.Segment_rate = slot->sample_rate;
        seg.Segment_reason.EmitReason_choice = reason;
        
        // Add chunk ID (1-based index)
        seg.Segment_chunk = chunk_idx + 1;
        
        // Populate points for this chunk dynamically
        seg.data_gyro.Datapoints_x_float32_count = TELEMETRY_CHUNK_SIZE;
        seg.data_gyro.Datapoints_y_float32_count = TELEMETRY_CHUNK_SIZE;
        seg.data_gyro.Datapoints_z_float32_count = TELEMETRY_CHUNK_SIZE;
        
        seg.data_accel.Datapoints_x_float32_count = TELEMETRY_CHUNK_SIZE;
        seg.data_accel.Datapoints_y_float32_count = TELEMETRY_CHUNK_SIZE;
        seg.data_accel.Datapoints_z_float32_count = TELEMETRY_CHUNK_SIZE;
        
        uint32_t start_offset = chunk_idx * TELEMETRY_CHUNK_SIZE;
        memcpy(seg.data_accel.Datapoints_x_float32, &slot->data.accel_x[start_offset], TELEMETRY_CHUNK_SIZE * sizeof(float));
        memcpy(seg.data_accel.Datapoints_y_float32, &slot->data.accel_y[start_offset], TELEMETRY_CHUNK_SIZE * sizeof(float));
        memcpy(seg.data_accel.Datapoints_z_float32, &slot->data.accel_z[start_offset], TELEMETRY_CHUNK_SIZE * sizeof(float));
        
        memcpy(seg.data_gyro.Datapoints_x_float32, &slot->data.gyro_x[start_offset], TELEMETRY_CHUNK_SIZE * sizeof(float));
        memcpy(seg.data_gyro.Datapoints_y_float32, &slot->data.gyro_y[start_offset], TELEMETRY_CHUNK_SIZE * sizeof(float));
        memcpy(seg.data_gyro.Datapoints_z_float32, &slot->data.gyro_z[start_offset], TELEMETRY_CHUNK_SIZE * sizeof(float));
        
        size_t encoded_len = 0;
        bool ok = cbor_encode_Segment(buf, max_cbor_len, &seg, &encoded_len);
        if (ok && global_mqtt_client && global_mqtt_client->is_connected(global_mqtt_client)) {
            char topic[128];
            snprintf(topic, sizeof(topic), "v1/%s/data/sensor", global_node_id);
            int msg_id = global_mqtt_client->publish(global_mqtt_client, topic, buf, encoded_len, 1, 0);
            
            if (msg_id < 0) {
                ESP_LOGW(TAG, "MQTT publish failed (msg_id=%d, outbox full). Retrying...", msg_id);
                int retries = 5;
                while (msg_id < 0 && retries > 0) {
                    vTaskDelay(pdMS_TO_TICKS(50));
                    msg_id = global_mqtt_client->publish(global_mqtt_client, topic, buf, encoded_len, 1, 0);
                    retries--;
                }

            }
            
            //this msg_id is different, gets updated above
            if (msg_id >= 0) {
                ESP_LOGI(TAG, "published segment %u chunk %u/%u (%zu bytes)", 
                         (unsigned int)slot->segment_id, (unsigned int)(chunk_idx + 1), (unsigned int)num_chunks, encoded_len);
            } else {
                ESP_LOGE(TAG, "failed to send segment %u chunk %u/%u (outbox overflow)", 
                         (unsigned int)slot->segment_id, (unsigned int)(chunk_idx + 1), (unsigned int)num_chunks);
                overall_ok = false;
            }
        } else {
            ESP_LOGE(TAG, "failed to encode segment %u chunk %u/%u", 
                     (unsigned int)slot->segment_id, (unsigned int)(chunk_idx + 1), (unsigned int)num_chunks);
            overall_ok = false;
        }
        
        // Brief yield between publishing chunks to avoid choking the MQTT transport layer
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    
    free(buf);
    return overall_ok;
}

static bool serialize_and_publish_inference(buffer_slot_t *slot, int32_t reason) {
    struct InferencePacket pkt;
    memset(&pkt, 0, sizeof(struct InferencePacket));
    
    pkt.InferencePacket_id = slot->segment_id;
    pkt.InferencePacket_reason.EmitReason_choice = reason;
    pkt.InferencePacket_r_m_id = slot->router_model_id;
    pkt.InferencePacket_ae_id = slot->active_submodel_id;
    pkt.InferencePacket_mse = slot->anomaly_score;
    pkt.InferencePacket_anom = slot->is_anomaly;
    
    uint8_t buf[256];
    size_t encoded_len = 0;
    bool ok = cbor_encode_InferencePacket(buf, sizeof(buf), &pkt, &encoded_len);
    if (ok && global_mqtt_client && global_mqtt_client->is_connected(global_mqtt_client)) {
        char topic[128];
        snprintf(topic, sizeof(topic), "v1/%s/inference", global_node_id);
        int msg_id = global_mqtt_client->publish(global_mqtt_client, topic, buf, encoded_len, 1, 0);
        
        if (msg_id < 0) {
            ESP_LOGW(TAG, "MQTT inference publish failed (msg_id=%d). Retrying...", msg_id);
            int retries = 5;
            while (msg_id < 0 && retries > 0) {
                vTaskDelay(pdMS_TO_TICKS(50));
                msg_id = global_mqtt_client->publish(global_mqtt_client, topic, buf, encoded_len, 1, 0);
                retries--;
            }
        }
        
        if (msg_id >= 0) {
            ESP_LOGI(TAG, "published inference packet for segment %u", (unsigned int)slot->segment_id);
        } else {
            ESP_LOGE(TAG, "failed to send inference packet for segment %u (outbox overflow)", (unsigned int)slot->segment_id);
            ok = false;
        }
    } else {
        ESP_LOGE(TAG, "failed to encode inference packet");
        ok = false;
    }
    
    return ok;
}

static void uploader_task(void *pvParameters) {
    quad_buffer_t *qb = (quad_buffer_t *)pvParameters;
    ESP_LOGI(TAG, "uploader task started");
    
    while (1) {
        int slot_idx;
        if (xQueueReceive(ready_send_queue, &slot_idx, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        
        buffer_slot_t *slot = &qb->slots[slot_idx];
        
        // wait for MQTT connection if not connected
        while (!global_mqtt_client || !global_mqtt_client->is_connected(global_mqtt_client)) {
            vTaskDelay(pdMS_TO_TICKS(100));
        }
        
        if (slot->flags & BUF_FLAG_READY_SEND) {
            int32_t reason = EmitReason_Continuous_m_c;
            bool should_send_raw = false;
            bool should_send_inf = slot->is_inferenced;
            
            if (slot->is_replay) {
                reason = EmitReason_ManualDump_m_c;
                should_send_raw = true;
            } else {
                if (global_stream_mode == StreamMode_Continuous_m_c) {
                    reason = EmitReason_Continuous_m_c;
                    should_send_raw = true;
                } else if (global_stream_mode == StreamMode_AnomalyOnly_m_c) {
                    if (slot->is_anomaly) {
                        reason = EmitReason_Anomaly_m_c;
                        should_send_raw = true;
                        should_send_inf = true;
                    } else {
                        should_send_raw = false;
                        should_send_inf = false;
                    }
                } else if (global_stream_mode == StreamMode_AnomalyOrPeriodic_m_c) {
                    if (slot->is_anomaly) {
                        reason = EmitReason_Anomaly_m_c;
                        should_send_raw = true;
                        should_send_inf = true;
                    } else if (global_cadence > 0 && (slot->segment_id % global_cadence == 0)) {
                        reason = EmitReason_Periodic_m_c;
                        should_send_raw = true;
                        should_send_inf = true;
                    } else {
                        should_send_raw = false;
                        should_send_inf = false;
                    }
                } else if (global_stream_mode == StreamMode_ContinuousScoreRawAnomaly_m_c) {
                    should_send_inf = slot->is_inferenced;
                    if (slot->is_anomaly) {
                        reason = EmitReason_Anomaly_m_c;
                        should_send_raw = true;
                    } else {
                        reason = EmitReason_Continuous_m_c;
                        should_send_raw = false;
                    }
                }
            }
            
            if (should_send_raw) {
                serialize_and_publish_segment(slot, reason);
            }
            
            if (should_send_inf) {
                serialize_and_publish_inference(slot, reason);
            }
            
            quad_buffer_lock(slot);
            slot->flags &= ~BUF_FLAG_READY_SEND;
            bool is_rep = slot->is_replay;
            slot->is_replay = false;
            bool is_free = (slot->flags & (BUF_FLAG_READY_INF | BUF_FLAG_READY_SEND | BUF_FLAG_PENDING_SD)) == 0;
            quad_buffer_unlock(slot);
            
            if (is_free) {
                xQueueSend(free_slots_queue, &slot_idx, 0);
                ESP_LOGI(TAG, "slot %d released back to free", slot_idx);
            }
        }
    }
}

void uploader_start(quad_buffer_t *qb) {
    xTaskCreatePinnedToCore(
        uploader_task,
        "uploader",
        8192,
        qb,
        3,
        &uploader_task_handle,
        0
    );
}
