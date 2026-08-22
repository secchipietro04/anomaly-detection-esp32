#ifndef MQTT_WRAPPER_H
#define MQTT_WRAPPER_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mqtt_wrapper_s mqtt_wrapper_t;

typedef void (*mqtt_topic_cb_t)(const uint8_t *data, size_t len, void *user_ctx);

typedef struct {
    const char *broker_uri;      //  "mqtt://192.168.1.100:1883"
    const char *client_id;       // optional, NULL for auto-generated

} mqtt_wrapper_config_t;

struct mqtt_wrapper_s {
    bool (*is_connected)(const mqtt_wrapper_t *self);
    int (*subscribe)(mqtt_wrapper_t *self, const char *topic, int qos, mqtt_topic_cb_t cb, void *user_ctx);
    int (*unsubscribe)(mqtt_wrapper_t *self, const char *topic);
    int (*publish)(mqtt_wrapper_t *self, const char *topic, const uint8_t *data, size_t len, int qos, int retain);
    void (*destroy)(mqtt_wrapper_t *self);
};


mqtt_wrapper_t* mqtt_wrapper_create(const mqtt_wrapper_config_t *config);

#ifdef __cplusplus
}
#endif

#endif // MQTT_WRAPPER_H