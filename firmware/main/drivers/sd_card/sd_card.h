#ifndef SD_CARD_H
#define SD_CARD_H

#include "esp_err.h"
#include "soc/gpio_num.h"
#include "hal/spi_types.h"
#include "driver/sdmmc_types.h"
#include "esp_err.h"
#include "sdmmc_cmd.h"
#include "driver/gpio.h"
#include "driver/spi_master.h"

#ifdef __cplusplus
extern "C" {
#endif

// default gpio pin and bus configuration
#define SD_CARD_DEFAULT_PIN_MISO  GPIO_NUM_42
#define SD_CARD_DEFAULT_PIN_MOSI  GPIO_NUM_41
#define SD_CARD_DEFAULT_PIN_CLK   GPIO_NUM_40
#define SD_CARD_DEFAULT_PIN_CS    GPIO_NUM_39
#define SD_CARD_DEFAULT_MOUNT_PT  "/sdcard"
#define SD_CARD_DEFAULT_SPI_HOST  SPI2_HOST
#define SD_CARD_DEFAULT_FREQ_KHZ  20000000 // 20 mhz
#define SD_CARD_DEFAULT_MAX_FILES 5
#define SD_CARD_DEFAULT_FORMAT_IF_MOUNT_FAILED false
/**
 * @brief sd card configuration for spi mount and vfs setup.
 */
typedef struct {
    const char *mount_point;
    gpio_num_t pin_miso;
    gpio_num_t pin_mosi;
    gpio_num_t pin_clk;
    gpio_num_t pin_cs;
    spi_host_device_t spi_host; // e.g., spi2_host or spi3_host
    int spi_freq_khz;           // frequency in khz (e.g., 20000 = 20mhz)
    uint8_t max_files;
    bool format_if_mount_failed;
} sd_card_config_t;

#define SD_CARD_DEFAULT_CONFIG() { \
    .mount_point = SD_CARD_DEFAULT_MOUNT_PT, \
    .pin_miso = SD_CARD_DEFAULT_PIN_MISO, \
    .pin_mosi = SD_CARD_DEFAULT_PIN_MOSI, \
    .pin_clk = SD_CARD_DEFAULT_PIN_CLK, \
    .pin_cs = SD_CARD_DEFAULT_PIN_CS, \
    .spi_host = SD_CARD_DEFAULT_SPI_HOST, \
    .spi_freq_khz = SD_CARD_DEFAULT_FREQ_KHZ, \
    .max_files = SD_CARD_DEFAULT_MAX_FILES, \
    .format_if_mount_failed = SD_CARD_DEFAULT_FORMAT_IF_MOUNT_FAILED \
}

/**
 * @brief runtime sd card object with mount and unmount operations.
 */
typedef struct sd_card sd_card_t;

struct sd_card {
    sd_card_config_t config;
    sdmmc_card_t *card_handle;
    bool is_mounted;

    esp_err_t (*mount)(sd_card_t *self);
    esp_err_t (*unmount)(sd_card_t *self);
};

/**
 * @brief initializes the sd card object.
 */
esp_err_t sd_card_init(sd_card_t *sd,const sd_card_config_t *config);

sd_card_config_t config;
#ifdef __cplusplus
}
#endif

#endif // SD_CARD_H