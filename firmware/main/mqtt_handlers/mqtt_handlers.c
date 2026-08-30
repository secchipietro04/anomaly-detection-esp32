#include "mqtt_handlers.h"
#include "main_functions.h"
#include "harvester.h"
#include "dump_reader.h"
#include "drivers/sd_card/sd_card.h"
#include "drivers/mems/ism330bx.h"
#include "cbor/gen/device.h"
#include "cbor/gen/ensemble.h"
#include "cbor/gen/telemetry_encode.h"
#include "cbor/gen/telemetry_decode.h"
#include "model_ensemble.h"
#include "tf_wrapper.h"
#include "esp_log.h"
#include "esp_system.h"
#include <string.h>

static const char *TAG = "mqtt_handlers";

static quad_buffer_t *local_qb = NULL;
static mqtt_wrapper_t *local_mqtt = NULL;

extern model_ensemble_t ensemble;

void publish_capabilities(void) {
    if (!local_mqtt || !local_mqtt->is_connected(local_mqtt)) return;
    
    struct NodeCapabilities caps;
    memset(&caps, 0, sizeof(caps));
    
    caps.NodeCapabilities_accel_freqs_float_count = 0;
    for (size_t i = 0; i < ISM330BX_ACCEL_FREQS_COUNT; i++) {
        caps.NodeCapabilities_accel_freqs_float[caps.NodeCapabilities_accel_freqs_float_count++] = ISM330BX_ACCEL_FREQS[i];
    }
    
    caps.NodeCapabilities_gyro_freqs_float_count = 0;
    for (size_t i = 0; i < ISM330BX_GYRO_FREQS_COUNT; i++) {
        caps.NodeCapabilities_gyro_freqs_float[caps.NodeCapabilities_gyro_freqs_float_count++] = ISM330BX_GYRO_FREQS[i];
    }
    
    // fetch registered tflite custom ops dynamically from wrapper exports
    size_t count = 0;
    const char** ops = tflite_micro_get_enabled_ops(&count);
    
    const size_t max_ops_capacity = sizeof(caps.NodeCapabilities_enabled_ops_tstr) / sizeof(caps.NodeCapabilities_enabled_ops_tstr[0]);
    caps.NodeCapabilities_enabled_ops_tstr_count = 0;
    for (size_t i = 0; i < count && caps.NodeCapabilities_enabled_ops_tstr_count < max_ops_capacity; i++) {
        size_t idx = caps.NodeCapabilities_enabled_ops_tstr_count++;
        caps.NodeCapabilities_enabled_ops_tstr[idx].value = (const uint8_t *)ops[i];
        caps.NodeCapabilities_enabled_ops_tstr[idx].len = strlen(ops[i]);
    }
    
    uint8_t buf[2048];
    size_t len = 0;
    if (cbor_encode_NodeCapabilities(buf, sizeof(buf), &caps, &len)) {
        char topic[128];
        snprintf(topic, sizeof(topic), "v1/%s/info/caps", global_node_id);
        local_mqtt->publish(local_mqtt, topic, buf, len, 1, 1);
        ESP_LOGI(TAG, "published capabilities telemetry once (ops count: %zu)", count);
    }
}

void publish_health_info(void) {
    if (!local_mqtt || !local_mqtt->is_connected(local_mqtt)) return;
    
    // publish static capabilities only once on startup connection
    static bool capabilities_published = false;
    if (!capabilities_published) {
        publish_capabilities();
        capabilities_published = true;
    }
    
    struct NodeHealthInfo health;
    memset(&health, 0, sizeof(struct NodeHealthInfo));
    
    health.NodeHealthInfo_ram = esp_get_free_heap_size();
    health.NodeHealthInfo_sd = global_sd_enabled;
    
    // cache_uint holds active model ids (router + memory + submodels) in health cbor telemetry
    const size_t max_cache_capacity = sizeof(health.NodeHealthInfo_cache_uint) / sizeof(health.NodeHealthInfo_cache_uint[0]);
    
    // count cached models and store in the static telemetry array
    health.NodeHealthInfo_cache_uint_count = 0;
    if (ensemble.router_model_loaded && health.NodeHealthInfo_cache_uint_count < max_cache_capacity) {
        health.NodeHealthInfo_cache_uint[health.NodeHealthInfo_cache_uint_count++] = ensemble.router_model.config.model_id;
    }
    if (ensemble.memory_model_loaded && health.NodeHealthInfo_cache_uint_count < max_cache_capacity) {
        health.NodeHealthInfo_cache_uint[health.NodeHealthInfo_cache_uint_count++] = ensemble.memory_model.config.model_id;
    }
    for (uint32_t i = 0; i < ensemble.submodel_cache.count && health.NodeHealthInfo_cache_uint_count < max_cache_capacity; i++) {
        health.NodeHealthInfo_cache_uint[health.NodeHealthInfo_cache_uint_count++] = ensemble.submodel_cache.items[i].model.config.model_id;
    }
    
    extern uint32_t global_last_segment_id;
    health.NodeHealthInfo_last = global_last_segment_id;
    
    // status zcbor_string
    health.NodeHealthInfo_status_present = true;
    if (global_dump_reading) {
        health.NodeHealthInfo_status.NodeHealthInfo_status.value = (const uint8_t *)"streaming_dump";
        health.NodeHealthInfo_status.NodeHealthInfo_status.len = strlen("streaming_dump");
    } else {
        health.NodeHealthInfo_status.NodeHealthInfo_status.value = (const uint8_t *)"idle";
        health.NodeHealthInfo_status.NodeHealthInfo_status.len = strlen("idle");
    }
    
    if (global_dump_time_ms > 0) {
        health.NodeHealthInfo_dump_t.NodeHealthInfo_dump_t = global_dump_time_ms;
        health.NodeHealthInfo_dump_t_present = true;
        
        health.NodeHealthInfo_dump_r.NodeHealthInfo_dump_r = global_dump_reading;
        health.NodeHealthInfo_dump_r_present = true;
        
        health.NodeHealthInfo_ips.NodeHealthInfo_ips = global_dump_ips;
        health.NodeHealthInfo_ips_present = true;
    }
    
    uint8_t buf[512];
    size_t len = 0;
    if (cbor_encode_NodeHealthInfo(buf, sizeof(buf), &health, &len)) {
        char topic[128];
        snprintf(topic, sizeof(topic), "v1/%s/info/health", global_node_id);
        local_mqtt->publish(local_mqtt, topic, buf, len, 1, 0);
        ESP_LOGI(TAG, "published health telemetry");
    }
}

static void cmd_reboot_cb(const uint8_t *data, size_t len, void *user_ctx) {
    ESP_LOGW(TAG, "reboot command trigger!");
    vTaskDelay(pdMS_TO_TICKS(500));
    esp_restart();
}

static void cmd_dump_cb(const uint8_t *data, size_t len, void *user_ctx) {
    ESP_LOGI(TAG, "received dump command");
    if (local_qb) {
        dump_reader_trigger(local_qb, false);
    }
}

static void cmd_dump_i_cb(const uint8_t *data, size_t len, void *user_ctx) {
    ESP_LOGI(TAG, "received dump_i command");
    if (local_qb) {
        dump_reader_trigger(local_qb, true);
    }
}

static void cmd_updt_status_cb(const uint8_t *data, size_t len, void *user_ctx) {
    ESP_LOGI(TAG, "received update status trigger");
    publish_health_info();
}

static void config_cb(const uint8_t *data, size_t len, void *user_ctx) {
    struct RuntimeConfig config;
    size_t decoded = 0;
    if (cbor_decode_RuntimeConfig(data, len, &config, &decoded)) {
        ESP_LOGI(TAG, "applied config. rate: %f, mode: %u", 
                 config.RuntimeConfig_rate, 
                 (unsigned int)config.RuntimeConfig_mode.StreamMode_choice);
        
        global_stream_mode = config.RuntimeConfig_mode.StreamMode_choice;
        global_sd_enabled = config.RuntimeConfig_sd_en;
        global_heartbeat_s = config.RuntimeConfig_beat;
        if (config.RuntimeConfig_cad_present) {
            global_cadence = config.RuntimeConfig_cad.RuntimeConfig_cad;
        }
        
        if (config.RuntimeConfig_rate != global_sample_rate) {
            harvester_update_rate(config.RuntimeConfig_rate);
        }
    }
}

static void ensemble_cb(const uint8_t *data, size_t len, void *user_ctx) {
    // payload_r is the top level union that can be a model pkg or ensemble config
    struct Payload_r payload;
    size_t decoded = 0;
    if (cbor_decode_Payload(data, len, &payload, &decoded)) {
        // user sent an ensemble config update (routing table + warmups)
        if (payload.Payload_choice == Payload_EnsembleConfig_m_c) {
            struct EnsembleConfig *cfg = &payload.Payload_EnsembleConfig_m;
            
            // compare old routes to identify evicted submodels
            for (uint32_t i = 0; i < ensemble.num_routes; i++) {
                uint32_t old_submodel_id = ensemble.routes[i].m_id;
                bool still_needed = false;
                
                // check if present in new routes
                for (size_t j = 0; j < cfg->EnsembleConfig_routes_RouteEntry_m_count; j++) {
                    if (cfg->EnsembleConfig_routes_RouteEntry_m[j].RouteEntry_m_id == old_submodel_id) {
                        still_needed = true;
                        break;
                    }
                }
                
                if (!still_needed) {
                    char path[64];
                    snprintf(path, sizeof(path), "/sdcard/models/model_%u.bin", (unsigned int)old_submodel_id);
                    sd_card_delete_file(path);
                    ESP_LOGI(TAG, "deleted evicted routing submodel %u from SD", (unsigned int)old_submodel_id);
                }
            }
            
            // map zcbor route structs to the internal engine route structure
            route_entry_t r_routes[MAX_ROUTES];
            for (size_t i = 0; i < cfg->EnsembleConfig_routes_RouteEntry_m_count && i < MAX_ROUTES; i++) {
                // routeentry_out_ix is mode index, routeentry_m_id is autoencoder ID
                r_routes[i].out_idx = cfg->EnsembleConfig_routes_RouteEntry_m[i].RouteEntry_out_ix;
                r_routes[i].m_id = cfg->EnsembleConfig_routes_RouteEntry_m[i].RouteEntry_m_id;
            }
            model_ensemble_set_routes(&ensemble, r_routes, cfg->EnsembleConfig_routes_RouteEntry_m_count);
            ensemble.warmup_steps = cfg->EnsembleConfig_warmup;
            
        } else if (payload.Payload_choice == Payload_ModelPackage_m_c) {
            // payload contains a specific model package (can be router, memory or autoencoder)
            struct ModelPackage_r *pkg = &payload.Payload_ModelPackage_m;
            
            if (pkg->ModelPackage_choice == RouterModelPackage_m_c) {
                // it is a router model. base contains the model metadata and data buffer
                struct RouterModelPackage *r_pkg = &pkg->RouterModelPackage_m;
                struct BaseModelPackage_r *base = &r_pkg->RouterModelPackage_BaseModelPackage_m;
                
                // delete old router if replacing it
                if (ensemble.router_model_loaded) {
                    uint32_t old_id = ensemble.router_model.config.model_id;
                    if (old_id != base->BaseModelPackage_m_id) {
                        char path[64];
                        snprintf(path, sizeof(path), "/sdcard/models/model_%u.bin", (unsigned int)old_id);
                        sd_card_delete_file(path);
                        ESP_LOGI(TAG, "deleted evicted router model %u from SD", (unsigned int)old_id);
                    }
                }
                
                // load the new router model into the ensemble
                model_ensemble_load_router(&ensemble, base->BaseModelPackage_data.value, base->BaseModelPackage_data.len);
                ensemble.router_model.config.model_id = base->BaseModelPackage_m_id;
                ensemble.router_model.config.archetype = ARCHETYPE_ROUTER;
                ensemble.router_model.config.temporal_depth = r_pkg->RouterModelPackage_tsteps;
                ensemble.router_model.config.frequency_bins = r_pkg->RouterModelPackage_accel_bins + r_pkg->RouterModelPackage_gyro_bins;
                ensemble.router_model.config.num_modes = r_pkg->RouterModelPackage_class;
                
            } else if (pkg->ModelPackage_choice == MemoryModelPackage_m_c) {
                // memory backbone model package. base contains the data buffer
                struct MemoryModelPackage *m_pkg = &pkg->MemoryModelPackage_m;
                struct BaseModelPackage_r *base = &m_pkg->MemoryModelPackage_BaseModelPackage_m;
                
                // delete old memory if replacing it
                if (ensemble.memory_model_loaded) {
                    uint32_t old_id = ensemble.memory_model.config.model_id;
                    if (old_id != base->BaseModelPackage_m_id) {
                        char path[64];
                        snprintf(path, sizeof(path), "/sdcard/models/model_%u.bin", (unsigned int)old_id);
                        sd_card_delete_file(path);
                        ESP_LOGI(TAG, "deleted evicted memory model %u from SD", (unsigned int)old_id);
                    }
                }
                
                model_ensemble_load_memory(&ensemble, base->BaseModelPackage_data.value, base->BaseModelPackage_data.len);
                ensemble.memory_model.config.model_id = base->BaseModelPackage_m_id;
                ensemble.memory_model.config.archetype = ARCHETYPE_MEMORY;
                ensemble.memory_model.config.frequency_bins = m_pkg->MemoryModelPackage_accel_bins + m_pkg->MemoryModelPackage_gyro_bins;
                ensemble.memory_model.config.d = m_pkg->MemoryModelPackage_state;
                ensemble.state_dim = m_pkg->MemoryModelPackage_state;
                
            } else if (pkg->ModelPackage_choice == AutoencoderModelPackage_m_c) {
                // autoencoder submodel package. base contains model ID and data buffer to save
                struct AutoencoderModelPackage *ae_pkg = &pkg->AutoencoderModelPackage_m;
                struct BaseModelPackage_r *base = &ae_pkg->AutoencoderModelPackage_BaseModelPackage_m;
                
                char path[64];
                snprintf(path, sizeof(path), "/sdcard/models/model_%u.bin", (unsigned int)base->BaseModelPackage_m_id);
                if (!sd_card_save_file(path, base->BaseModelPackage_data.value, base->BaseModelPackage_data.len)) {
                    ESP_LOGE(TAG, "failed to save submodel %u to SD", (unsigned int)base->BaseModelPackage_m_id);
                    global_sd_enabled = false;
                }
            }
        }
    }
}

void mqtt_handlers_init(mqtt_wrapper_t *mqtt_client, quad_buffer_t *qb) {
    local_mqtt = mqtt_client;
    local_qb = qb;
    
    if (local_mqtt) {
        char topic[128];
        
        snprintf(topic, sizeof(topic), "v1/%s/cmd/reboot", global_node_id);
        local_mqtt->subscribe(local_mqtt, topic, 1, cmd_reboot_cb, NULL);
        
        snprintf(topic, sizeof(topic), "v1/%s/cmd/dump", global_node_id);
        local_mqtt->subscribe(local_mqtt, topic, 1, cmd_dump_cb, NULL);
        
        snprintf(topic, sizeof(topic), "v1/%s/cmd/dump_i", global_node_id);
        local_mqtt->subscribe(local_mqtt, topic, 1, cmd_dump_i_cb, NULL);
        
        snprintf(topic, sizeof(topic), "v1/%s/cmd/updt_status", global_node_id);
        local_mqtt->subscribe(local_mqtt, topic, 1, cmd_updt_status_cb, NULL);
        
        snprintf(topic, sizeof(topic), "v1/%s/config", global_node_id);
        local_mqtt->subscribe(local_mqtt, topic, 1, config_cb, NULL);
        
        snprintf(topic, sizeof(topic), "v1/%s/ensemble", global_node_id);
        local_mqtt->subscribe(local_mqtt, topic, 1, ensemble_cb, NULL);
        
        snprintf(topic, sizeof(topic), "v1/%s/models/router/+", global_node_id);
        local_mqtt->subscribe(local_mqtt, topic, 1, ensemble_cb, NULL);
        
        snprintf(topic, sizeof(topic), "v1/%s/models/memory/+", global_node_id);
        local_mqtt->subscribe(local_mqtt, topic, 1, ensemble_cb, NULL);
        
        snprintf(topic, sizeof(topic), "v1/%s/models/submodel/+", global_node_id);
        local_mqtt->subscribe(local_mqtt, topic, 1, ensemble_cb, NULL);
    }
}
