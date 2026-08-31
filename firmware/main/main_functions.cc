#include "main_functions.h"
#include "nvs_flash.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_mac.h"
#include <string.h>
#include <stdlib.h>

extern "C" {
#include "quad_buffer.h"
#include "harvester.h"
#include "inferencer.h"
#include "sd_writer.h"
#include "uploader.h"
#include "mqtt_handlers.h"
#include "drivers/sd_card/sd_card.h"
#include "network/wifi/wifi.h"
#include "network/mqtt/mqtt.h"
}

static const char *TAG = "main_app";

#define DEFAULT_BROKER "mqtt://10.49.79.1:1883"

// global configurations definitions
float global_sample_rate = 3840.0f;
bool global_sd_enabled = true;
bool global_inference_enabled = true;
uint32_t global_stream_mode = 1; // StreamMode_Continuous
uint32_t global_cadence = 10;     // default mixed cadence
uint32_t global_skip_amount = 0;
uint32_t global_heartbeat_s = 5;  // heartbeat loop interval

// statistics globals
uint32_t global_dump_time_ms = 0;
bool global_dump_reading = false;
float global_dump_ips = 0.0f;
uint32_t global_last_segment_id = 0;

// task handles
TaskHandle_t harvester_task_handle = NULL;
TaskHandle_t sd_writer_task_handle = NULL;

// node ID
char global_node_id[32] = "esp32_device";

static sd_card_t sd_card;
static wifi_wrapper_t *wifi_client = NULL;
mqtt_wrapper_t *global_mqtt_client = NULL;

// dynamically allocate quad buffer to prevent DRAM compile-time overflow
static quad_buffer_t *qb = NULL;

void setup(void) {
    // initialize non volatile storage
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    // mount storage FAT filesystem
    sd_card_init(&sd_card, NULL);
    esp_err_t err = sd_card.mount(&sd_card);
    if (err != ESP_OK) {
        global_sd_enabled = false;
        ESP_LOGE(TAG, "failed mounting FAT filesystem on SD card.");
    }
    
    // query STA interface mac address
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    snprintf(global_node_id, sizeof(global_node_id), "%02x%02x%02x%02x%02x%02x",
              mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    ESP_LOGI(TAG, "system node ID derived from WiFi MAC: %s", global_node_id);
    
    // start wifi link ( NVS bootstrapper handles credentials )
    wifi_client = wifi_wrapper_create("wifi");
    if (wifi_client) {
        wifi_client->connect(wifi_client);
    }
    
    // start mqtt broker client
    mqtt_wrapper_config_t mqtt_cfg = {
        .broker_uri = DEFAULT_BROKER,
        .client_id = global_node_id
    };
    global_mqtt_client = mqtt_wrapper_create(&mqtt_cfg);
    
    // allocate and initialize the data staging structure on the heap
    qb = (quad_buffer_t *)calloc(1, sizeof(quad_buffer_t));
    if (!qb) {
        ESP_LOGE(TAG, "failed to allocate memory for quad buffer staging structure!");
        return;
    }
    quad_buffer_init(qb);
    
    // register command handler callbacks
    mqtt_handlers_init(global_mqtt_client, qb);
    
    // spawn concurrent operational threads
    harvester_start(qb);
    inferencer_start(qb);
    sd_writer_start(qb);
    uploader_start(qb);
}

void loop(void) {
    // publish heartbeat and current system health
    publish_health_info();
    vTaskDelay(pdMS_TO_TICKS(global_heartbeat_s * 1000));
}