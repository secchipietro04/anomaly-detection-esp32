#include "mqtt.h"
#include <stdlib.h>
#include <string.h>
#include "esp_log.h"
#include "mqtt_client.h"

#define MAX_TOPIC_ROUTES 10

typedef struct {
    char topic[64];
    mqtt_topic_cb_t callback;
    void *user_ctx;
} mqtt_route_t;

typedef struct {
    mqtt_wrapper_t interface;
    esp_mqtt_client_handle_t client;
    bool is_connected;
    mqtt_route_t routes[MAX_TOPIC_ROUTES];
    int route_count;
} mqtt_wrapper_impl_t;

static const char *TAG = "MQTT_WRAPPER";

static bool is_connected_impl(const mqtt_wrapper_t *self) {
    if (!self) return false;
    const mqtt_wrapper_impl_t *impl = (const mqtt_wrapper_impl_t *)self;
    return impl->is_connected;
}

static int subscribe_impl(mqtt_wrapper_t *self, const char *topic, int qos, mqtt_topic_cb_t cb, void *user_ctx) {
    if (!self) return -1;
    mqtt_wrapper_impl_t *impl = (mqtt_wrapper_impl_t *)self;

    if (!impl->is_connected || !impl->client || impl->route_count >= MAX_TOPIC_ROUTES) {
        return -1;
    }

    for (int i = 0; i < impl->route_count; i++) {
        if (strcmp(impl->routes[i].topic, topic) == 0) {
            impl->routes[i].callback = cb;
            impl->routes[i].user_ctx = user_ctx;
            return esp_mqtt_client_subscribe(impl->client, topic, qos);
        }
    }

    strncpy(impl->routes[impl->route_count].topic, topic, sizeof(impl->routes[impl->route_count].topic) - 1);
    impl->routes[impl->route_count].topic[sizeof(impl->routes[impl->route_count].topic) - 1] = '\0';
    impl->routes[impl->route_count].callback = cb;
    impl->routes[impl->route_count].user_ctx = user_ctx;
    impl->route_count++;

    return esp_mqtt_client_subscribe(impl->client, topic, qos);
}

static int unsubscribe_impl(mqtt_wrapper_t *self, const char *topic) {
    if (!self) return -1;
    mqtt_wrapper_impl_t *impl = (mqtt_wrapper_impl_t *)self;

    if (!impl->is_connected || !impl->client) {
        return -1;
    }

    int found_index = -1;
    for (int i = 0; i < impl->route_count; i++) {
        if (strcmp(impl->routes[i].topic, topic) == 0) {
            found_index = i;
            break;
        }
    }

    if (found_index != -1) {
        for (int i = found_index; i < impl->route_count - 1; i++) {
            impl->routes[i] = impl->routes[i + 1];
        }
        impl->route_count--;
    }

    return esp_mqtt_client_unsubscribe(impl->client, topic);
}

static int publish_impl(mqtt_wrapper_t *self, const char *topic, const uint8_t *data, size_t len, int qos, int retain) {
    if (!self) return -1;
    mqtt_wrapper_impl_t *impl = (mqtt_wrapper_impl_t *)self;

    if (!impl->is_connected || !impl->client) {
        return -1;
    }

    return esp_mqtt_client_publish(impl->client, topic, (const char *)data, (int)len, qos, retain);
}

static void destroy_impl(mqtt_wrapper_t *self) {
    if (!self) return;
    mqtt_wrapper_impl_t *impl = (mqtt_wrapper_impl_t *)self;

    if (impl->client) {
        esp_mqtt_client_stop(impl->client);
        esp_mqtt_client_destroy(impl->client);
    }
    free(impl);
}

static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    mqtt_wrapper_impl_t *impl = (mqtt_wrapper_impl_t *)handler_args;
    if (!impl) return;

    esp_mqtt_event_handle_t event = event_data;

    switch ((esp_mqtt_event_id_t)event_id) {
        case MQTT_EVENT_CONNECTED:
            impl->is_connected = true;
            ESP_LOGI(TAG, "MQTT Connected");
            break;

        case MQTT_EVENT_DISCONNECTED:
            impl->is_connected = false;
            ESP_LOGW(TAG, "MQTT Disconnected");
            break;

        case MQTT_EVENT_DATA:
            for (int i = 0; i < impl->route_count; i++) {
                if (strlen(impl->routes[i].topic) == event->topic_len &&
                    strncmp(impl->routes[i].topic, event->topic, event->topic_len) == 0) {
                    
                    if (impl->routes[i].callback) {
                        impl->routes[i].callback(
                            (const uint8_t *)event->data,
                            (size_t)event->data_len,
                            impl->routes[i].user_ctx
                        );
                    }
                    break;
                }
            }
            break;

        case MQTT_EVENT_ERROR:
            ESP_LOGE(TAG, "MQTT Error event");
            break;

        default:
            break;
    }
}

mqtt_wrapper_t* mqtt_wrapper_create(const mqtt_wrapper_config_t *config) {
    if (!config || !config->broker_uri) {
        ESP_LOGE(TAG, "Invalid configuration");
        return NULL;
    }

    mqtt_wrapper_impl_t *impl = (mqtt_wrapper_impl_t *)calloc(1, sizeof(mqtt_wrapper_impl_t));
    if (!impl) {
        ESP_LOGE(TAG, "Memory allocation failed");
        return NULL;
    }

    // Bind functions to struct pointers
    impl->interface.is_connected = is_connected_impl;
    impl->interface.subscribe    = subscribe_impl;
    impl->interface.unsubscribe  = unsubscribe_impl;
    impl->interface.publish      = publish_impl;
    impl->interface.destroy      = destroy_impl;

    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = config->broker_uri,
    };

    if (config->client_id) {
        mqtt_cfg.credentials.client_id = config->client_id;
    }

    

    impl->client = esp_mqtt_client_init(&mqtt_cfg);
    if (!impl->client) {
        ESP_LOGE(TAG, "Failed to initialize MQTT client");
        free(impl);
        return NULL;
    }

    esp_mqtt_client_register_event(impl->client, ESP_EVENT_ANY_ID, mqtt_event_handler, impl);

    esp_err_t err = esp_mqtt_client_start(impl->client);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start MQTT client: %s", esp_err_to_name(err));
        esp_mqtt_client_destroy(impl->client);
        free(impl);
        return NULL;
    }

    return &impl->interface;
}