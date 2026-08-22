#ifndef WIFI_WRAPPER_H
#define WIFI_WRAPPER_H

#include "esp_err.h"
#include "esp_wifi.h"
#include "wifi.h"
typedef struct wifi_wrapper_t wifi_wrapper_t;

typedef struct {
    char ssid[32];
    char password[64];
    EventGroupHandle_t event_group;
    esp_netif_t *netif;
    int retry_count;
    int max_retries;
} wifi_private_t;

struct wifi_wrapper_t {
    wifi_private_t *data; //  internal state 

    esp_err_t (*connect)(wifi_wrapper_t *self);
    void (*destroy)(wifi_wrapper_t *self);
};

/**
 * Constructor: Allocates instance and populates method function pointers.
 */
wifi_wrapper_t* wifi_wrapper_create(const char *nvs_namespace);

#endif // WIFI_WRAPPER_H