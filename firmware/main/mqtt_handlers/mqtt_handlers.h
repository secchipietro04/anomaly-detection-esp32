#ifndef MQTT_HANDLERS_H
#define MQTT_HANDLERS_H

#include "quad_buffer.h"
#include "network/mqtt/mqtt.h"

#ifdef __cplusplus
extern "C" {
#endif

// register subscriptions
void mqtt_handlers_init(mqtt_wrapper_t *mqtt_client, quad_buffer_t *qb);

// publish health
void publish_health_info(void);

#ifdef __cplusplus
}
#endif

#endif // MQTT_HANDLERS_H
