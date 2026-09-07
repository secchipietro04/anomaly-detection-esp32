#ifndef SD_CARD_SPI_H
#define SD_CARD_SPI_H

#include "esp_err.h"
#include "soc/gpio_num.h"
#include "hal/spi_types.h"
#include "sdmmc_cmd.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Legacy SPI pinout and bus defaults
#define SD_CARD_SPI_DEFAULT_PIN_MISO  GPIO_NUM_42
#define SD_CARD_SPI_DEFAULT_PIN_MOSI  GPIO_NUM_41
#define SD_CARD_SPI_DEFAULT_PIN_CLK   GPIO_NUM_40
#define SD_CARD_SPI_DEFAULT_PIN_CS    GPIO_NUM_39
#define SD_CARD_SPI_DEFAULT_HOST      SPI2_HOST
#define SD_CARD_SPI_DEFAULT_FREQ_KHZ  20000 // 20 MHz

typedef struct {
    const char *mount_point;
    gpio_num_t pin_miso;
    gpio_num_t pin_mosi;
    gpio_num_t pin_clk;
    gpio_num_t pin_cs;
    spi_host_device_t spi_host;
    int spi_freq_khz;
    uint8_t max_files;
    bool format_if_mount_failed;
    size_t allocation_unit_size;
} sd_card_spi_config_t;

#define SD_CARD_SPI_DEFAULT_CONFIG() { \
    .mount_point = "/sdcard", \
    .pin_miso = SD_CARD_SPI_DEFAULT_PIN_MISO, \
    .pin_mosi = SD_CARD_SPI_DEFAULT_PIN_MOSI, \
    .pin_clk = SD_CARD_SPI_DEFAULT_PIN_CLK, \
    .pin_cs = SD_CARD_SPI_DEFAULT_PIN_CS, \
    .spi_host = SD_CARD_SPI_DEFAULT_HOST, \
    .spi_freq_khz = SD_CARD_SPI_DEFAULT_FREQ_KHZ, \
    .max_files = 5, \
    .format_if_mount_failed = false, \
    .allocation_unit_size = 16 * 1024 \
}

esp_err_t sd_card_spi_mount(const sd_card_spi_config_t *config, sdmmc_card_t **out_card);
esp_err_t sd_card_spi_unmount(const char *mount_point, sdmmc_card_t *card);

#ifdef __cplusplus
}
#endif

#endif // SD_CARD_SPI_H
