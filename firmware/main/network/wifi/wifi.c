#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "wifi.h"

#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1

static const char *TAG = "WIFI_WRAPPER";



static void event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data) {
    wifi_wrapper_t *self = (wifi_wrapper_t *)arg;
    wifi_private_t *priv = (wifi_private_t *)self->data;

    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (priv->retry_count < priv->max_retries) {
            esp_wifi_connect();
            priv->retry_count++;
            ESP_LOGI(TAG, "Retrying connection to AP (%d/%d)...", priv->retry_count, priv->max_retries);
        } else {
            xEventGroupSetBits(priv->event_group, WIFI_FAIL_BIT);
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        priv->retry_count = 0;
        xEventGroupSetBits(priv->event_group, WIFI_CONNECTED_BIT);
    }
}

static esp_err_t load_nvs_credentials(wifi_private_t *priv, const char *nvs_namespace) {
    nvs_handle_t handle;
    esp_err_t err = nvs_open(nvs_namespace, NVS_READONLY, &handle);
    if (err != ESP_OK) return err;

    size_t ssid_len = sizeof(priv->ssid);
    size_t pass_len = sizeof(priv->password);

    err = nvs_get_str(handle, "ssid", priv->ssid, &ssid_len);
    if (err == ESP_OK) {
        err = nvs_get_str(handle, "password", priv->password, &pass_len);
    }

    nvs_close(handle);
    return err;
}

static esp_err_t wifi_connect_impl(wifi_wrapper_t *self) {
    if (!self || !self->data) return ESP_ERR_INVALID_ARG;
    wifi_private_t *priv = (wifi_private_t *)self->data;

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    priv->netif = esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id, instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, self, &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, self, &instance_got_ip));

    wifi_config_t wifi_config = {
        .sta = {
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    strlcpy((char *)wifi_config.sta.ssid, priv->ssid, sizeof(wifi_config.sta.ssid));
    strlcpy((char *)wifi_config.sta.password, priv->password, sizeof(wifi_config.sta.password));

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    EventBits_t bits = xEventGroupWaitBits(
        priv->event_group,
        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
        pdFALSE,
        pdFALSE,
        portMAX_DELAY
    );

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Connected to SSID: %s", priv->ssid);
        return ESP_OK;
    }

    ESP_LOGE(TAG, "Failed connection to SSID: %s", priv->ssid);
    return ESP_FAIL;
}

static void wifi_destroy_impl(wifi_wrapper_t *self) {
    if (!self) return;

    if (self->data) {
        wifi_private_t *priv = (wifi_private_t *)self->data;
        esp_wifi_stop();
        esp_wifi_deinit();
        if (priv->netif) esp_netif_destroy(priv->netif);
        if (priv->event_group) vEventGroupDelete(priv->event_group);
        free(priv);
    }
    free(self);
}

wifi_wrapper_t* wifi_wrapper_create(const char *nvs_namespace) {
    wifi_wrapper_t *self = calloc(1, sizeof(wifi_wrapper_t));
    wifi_private_t *priv = calloc(1, sizeof(wifi_private_t));
    if (!self || !priv) {
        free(self);
        free(priv);
        return NULL;
    }

    priv->max_retries = 5;
    priv->event_group = xEventGroupCreate();

    if (load_nvs_credentials(priv, nvs_namespace) != ESP_OK) {
        vEventGroupDelete(priv->event_group);
        free(priv);
        free(self);
        return NULL;
    }

    // Bind state and internal functions to public struct pointers
    self->data = priv;
    self->connect = wifi_connect_impl;
    self->destroy = wifi_destroy_impl;

    return self;
}