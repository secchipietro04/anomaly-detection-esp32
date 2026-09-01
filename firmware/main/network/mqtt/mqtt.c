#include "mqtt.h"
#include <stdlib.h>
#include <string.h>
#include "esp_log.h"
#include "mqtt_client.h"

#define MAX_TOPIC_ROUTES 32

typedef struct {
    char topic[128];
    mqtt_topic_cb_t callback;
    void *user_ctx;
} mqtt_route_t;

typedef struct {
    mqtt_wrapper_t interface;
    esp_mqtt_client_handle_t client;
    bool is_connected;
    mqtt_route_t routes[MAX_TOPIC_ROUTES];
    int route_count;
    char current_topic[128];
    uint8_t *msg_buffer;
    size_t msg_received_len;
    size_t msg_total_len;
} mqtt_wrapper_impl_t;

static const char *TAG = "MQTT_WRAPPER";

static bool is_connected_impl(const mqtt_wrapper_t *self) {
    if (!self) return false;
    const mqtt_wrapper_impl_t *impl = (const mqtt_wrapper_impl_t *)self;
    return impl->is_connected;
}

static bool mqtt_topic_match(const char *pattern, const char *topic, size_t topic_len) {
    char topic_null[128];
    if (topic_len >= sizeof(topic_null)) return false;
    memcpy(topic_null, topic, topic_len);
    topic_null[topic_len] = '\0';

    const char *p = pattern;
    const char *t = topic_null;

    while (*p && *t) {
        if (*p == '+') {
            p++;
            while (*t && *t != '/') t++;
        } else if (*p == '#') {
            return true;
        } else if (*p == *t) {
            p++;
            t++;
        } else {
            return false;
        }
    }
    if (*p == '+') p++;
    if (*p == '#') return true;
    return (*p == '\0' && *t == '\0');
}

static int subscribe_impl(mqtt_wrapper_t *self, const char *topic, int qos, mqtt_topic_cb_t cb, void *user_ctx) {
    if (!self || !topic) return -1;
    mqtt_wrapper_impl_t *impl = (mqtt_wrapper_impl_t *)self;

    if (impl->route_count >= MAX_TOPIC_ROUTES) {
        ESP_LOGE(TAG, "Max MQTT routes (%d) exceeded", MAX_TOPIC_ROUTES);
        return -1;
    }

    int route_idx = -1;
    for (int i = 0; i < impl->route_count; i++) {
        if (strcmp(impl->routes[i].topic, topic) == 0) {
            route_idx = i;
            break;
        }
    }

    if (route_idx == -1) {
        route_idx = impl->route_count++;
        strncpy(impl->routes[route_idx].topic, topic, sizeof(impl->routes[route_idx].topic) - 1);
        impl->routes[route_idx].topic[sizeof(impl->routes[route_idx].topic) - 1] = '\0';
    }

    impl->routes[route_idx].callback = cb;
    impl->routes[route_idx].user_ctx = user_ctx;

    if (impl->is_connected && impl->client) {
        return esp_mqtt_client_subscribe(impl->client, topic, qos);
    }
    return 0;
}

static int unsubscribe_impl(mqtt_wrapper_t *self, const char *topic) {
    if (!self || !topic) return -1;
    mqtt_wrapper_impl_t *impl = (mqtt_wrapper_impl_t *)self;

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

    if (impl->is_connected && impl->client) {
        return esp_mqtt_client_unsubscribe(impl->client, topic);
    }
    return 0;
}

static int publish_impl(mqtt_wrapper_t *self, const char *topic, const uint8_t *data, size_t len, int qos, int retain) {
    if (!self || !topic) return -1;
    mqtt_wrapper_impl_t *impl = (mqtt_wrapper_impl_t *)self;

    if (!impl->is_connected || !impl->client) {
        return -1;
    }

    return esp_mqtt_client_publish(impl->client, topic, (const char *)data, (int)len, qos, retain);
}

static void destroy_impl(mqtt_wrapper_t *self) {
    if (!self) return;
    mqtt_wrapper_impl_t *impl = (mqtt_wrapper_impl_t *)self;

    if (impl->msg_buffer) {
        free(impl->msg_buffer);
        impl->msg_buffer = NULL;
    }
    if (impl->client) {
        esp_mqtt_client_stop(impl->client);
        esp_mqtt_client_destroy(impl->client);
    }
    free(impl);
}

static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    mqtt_wrapper_impl_t *impl = (mqtt_wrapper_impl_t *)handler_args;
    if (!impl) return;

    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;

    switch ((esp_mqtt_event_id_t)event_id) {
        case MQTT_EVENT_CONNECTED:
            impl->is_connected = true;
            ESP_LOGI(TAG, "MQTT Connected. Subscribing to %d registered topics...", impl->route_count);
            for (int i = 0; i < impl->route_count; i++) {
                esp_mqtt_client_subscribe(impl->client, impl->routes[i].topic, 1);
            }
            break;

        case MQTT_EVENT_DISCONNECTED:
            impl->is_connected = false;
            ESP_LOGW(TAG, "MQTT Disconnected");
            if (impl->msg_buffer) {
                free(impl->msg_buffer);
                impl->msg_buffer = NULL;
            }
            impl->msg_received_len = 0;
            impl->msg_total_len = 0;
            break;

        case MQTT_EVENT_DATA:
            if (event->current_data_offset == 0) {
                if (event->topic && event->topic_len > 0 && event->topic_len < sizeof(impl->current_topic)) {
                    memcpy(impl->current_topic, event->topic, event->topic_len);
                    impl->current_topic[event->topic_len] = '\0';
                }
                impl->msg_total_len = event->total_data_len;
                impl->msg_received_len = 0;
                if (impl->msg_buffer) {
                    free(impl->msg_buffer);
                    impl->msg_buffer = NULL;
                }
                impl->msg_buffer = (uint8_t *)malloc(event->total_data_len);
            }

            if (impl->msg_buffer && (impl->msg_received_len + event->data_len <= impl->msg_total_len)) {
                memcpy(impl->msg_buffer + event->current_data_offset, event->data, event->data_len);
                impl->msg_received_len += event->data_len;
            }

            if (impl->msg_buffer && (impl->msg_received_len >= impl->msg_total_len)) {
                for (int i = 0; i < impl->route_count; i++) {
                    if (mqtt_topic_match(impl->routes[i].topic, impl->current_topic, strlen(impl->current_topic))) {
                        if (impl->routes[i].callback) {
                            impl->routes[i].callback(
                                impl->msg_buffer,
                                impl->msg_total_len,
                                impl->routes[i].user_ctx
                            );
                        }
                    }
                }
                free(impl->msg_buffer);
                impl->msg_buffer = NULL;
                impl->msg_received_len = 0;
                impl->msg_total_len = 0;
            }
            break;

        case MQTT_EVENT_ERROR:
            ESP_LOGE(TAG, "MQTT Error event");
            break;

        default:
            break;
    }
}

mqtt_wrapper_t *mqtt_wrapper_create(const mqtt_wrapper_config_t *config) {
    if (!config || !config->broker_uri) return NULL;

    mqtt_wrapper_impl_t *impl = (mqtt_wrapper_impl_t *)calloc(1, sizeof(mqtt_wrapper_impl_t));
    if (!impl) return NULL;

    impl->interface.is_connected = is_connected_impl;
    impl->interface.subscribe = subscribe_impl;
    impl->interface.unsubscribe = unsubscribe_impl;
    impl->interface.publish = publish_impl;
    impl->interface.destroy = destroy_impl;
    impl->route_count = 0;
    impl->msg_buffer = NULL;

    esp_mqtt_client_config_t mqtt_cfg = {
        .broker = {
            .address = {
                .uri = config->broker_uri
            }
        },
        .credentials = {
            .client_id = config->client_id
        },
        .buffer = {
            .size = 4096,
            .out_size = 4096
        }
    };

    impl->client = esp_mqtt_client_init(&mqtt_cfg);
    if (!impl->client) {
        free(impl);
        return NULL;
    }

    esp_mqtt_client_register_event(impl->client, (esp_mqtt_event_id_t)ESP_EVENT_ANY_ID, mqtt_event_handler, impl);
    esp_mqtt_client_start(impl->client);

    return (mqtt_wrapper_t *)impl;
}
