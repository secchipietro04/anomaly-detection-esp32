#ifndef SD_CARD_SDIO_H
#define SD_CARD_SDIO_H

#include "esp_err.h"
#include "soc/gpio_num.h"
#include "sdmmc_cmd.h"
#include "driver/sdmmc_host.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Adafruit 4682 SDIO 4-bit pinout (ESP32-S3 Header J1 contiguous, no wire crossing)
// Pin 21 -> CLK  (GPIO 13)
// Pin 20 -> DAT0 (GPIO 12)
// Pin 19 -> CMD  (GPIO 11)
// Pin 18 -> DAT3 (GPIO 10)
// Pin 17 -> DAT1 (GPIO 9)
// Pin 16 -> DAT2 (GPIO 46)
// Pin 15 -> DET  (GPIO 3, bypassato per affidabilità)
#define SD_CARD_SDIO_DEFAULT_PIN_CLK   GPIO_NUM_13
#define SD_CARD_SDIO_DEFAULT_PIN_D0    GPIO_NUM_12
#define SD_CARD_SDIO_DEFAULT_PIN_CMD   GPIO_NUM_11
#define SD_CARD_SDIO_DEFAULT_PIN_D3    GPIO_NUM_10
#define SD_CARD_SDIO_DEFAULT_PIN_D1    GPIO_NUM_9
#define SD_CARD_SDIO_DEFAULT_PIN_D2    GPIO_NUM_46
#define SD_CARD_SDIO_DEFAULT_PIN_CD    SDMMC_SLOT_NO_CD
#define SD_CARD_SDIO_DEFAULT_FREQ_KHZ  SDMMC_FREQ_HIGHSPEED // 40 MHz

typedef struct {
    const char *mount_point;
    gpio_num_t pin_clk;
    gpio_num_t pin_cmd;
    gpio_num_t pin_d0;
    gpio_num_t pin_d1;
    gpio_num_t pin_d2;
    gpio_num_t pin_d3;
    gpio_num_t pin_cd;
    int freq_khz;
    uint8_t max_files;
    bool format_if_mount_failed;
    size_t allocation_unit_size;
} sd_card_sdio_config_t;

#define SD_CARD_SDIO_DEFAULT_CONFIG() { \
    .mount_point = "/sdcard", \
    .pin_clk = SD_CARD_SDIO_DEFAULT_PIN_CLK, \
    .pin_cmd = SD_CARD_SDIO_DEFAULT_PIN_CMD, \
    .pin_d0  = SD_CARD_SDIO_DEFAULT_PIN_D0, \
    .pin_d1  = SD_CARD_SDIO_DEFAULT_PIN_D1, \
    .pin_d2  = SD_CARD_SDIO_DEFAULT_PIN_D2, \
    .pin_d3  = SD_CARD_SDIO_DEFAULT_PIN_D3, \
    .pin_cd  = SD_CARD_SDIO_DEFAULT_PIN_CD, \
    .freq_khz = SD_CARD_SDIO_DEFAULT_FREQ_KHZ, \
    .max_files = 5, \
    .format_if_mount_failed = false, \
    .allocation_unit_size = 64 * 1024 \
}

esp_err_t sd_card_sdio_mount(const sd_card_sdio_config_t *config, sdmmc_card_t **out_card);
esp_err_t sd_card_sdio_unmount(const char *mount_point, sdmmc_card_t *card);

#ifdef __cplusplus
}
#endif

#endif // SD_CARD_SDIO_H
