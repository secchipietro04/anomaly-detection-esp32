#ifndef SD_CARD_H
#define SD_CARD_H

#include "esp_err.h"
#include "soc/gpio_num.h"
#include "hal/spi_types.h"
#include "driver/sdmmc_types.h"
#include "sdmmc_cmd.h"
#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "sd_card_sdio.h"
#include "sd_card_spi.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SD_CARD_BUS_SDIO_4BIT = 0, // Adafruit 4682 in SDIO 4-bit mode (Default, High Performance)
    SD_CARD_BUS_SPI            // Legacy SPI mode
} sd_card_bus_type_t;

// Common defaults
#define SD_CARD_DEFAULT_MOUNT_PT  "/sdcard"
#define SD_CARD_DEFAULT_MAX_FILES 5
#define SD_CARD_DEFAULT_FORMAT_IF_MOUNT_FAILED false
#define SD_CARD_DEFAULT_ALLOC_UNIT (64 * 1024) // 64 KB cluster

/**
 * @brief Unified SD card configuration supporting both SDIO 4-bit and legacy SPI.
 */
typedef struct {
    const char *mount_point;
    sd_card_bus_type_t bus_type;
    uint8_t max_files;
    bool format_if_mount_failed;
    size_t allocation_unit_size;

    // SDIO 4-Bit Configuration (Default)
    sd_card_sdio_config_t sdio;

    // Legacy SPI Configuration
    sd_card_spi_config_t spi;

    // Flat legacy SPI fields kept for 100% backward compatibility
    gpio_num_t pin_miso;
    gpio_num_t pin_mosi;
    gpio_num_t pin_clk;
    gpio_num_t pin_cs;
    spi_host_device_t spi_host;
    int spi_freq_khz;
} sd_card_config_t;

// Default configuration uses Adafruit SDIO 4-Bit mode
#define SD_CARD_DEFAULT_CONFIG() { \
    .mount_point = SD_CARD_DEFAULT_MOUNT_PT, \
    .bus_type = SD_CARD_BUS_SDIO_4BIT, \
    .max_files = SD_CARD_DEFAULT_MAX_FILES, \
    .format_if_mount_failed = SD_CARD_DEFAULT_FORMAT_IF_MOUNT_FAILED, \
    .allocation_unit_size = SD_CARD_DEFAULT_ALLOC_UNIT, \
    .sdio = SD_CARD_SDIO_DEFAULT_CONFIG(), \
    .spi = SD_CARD_SPI_DEFAULT_CONFIG(), \
    .pin_miso = SD_CARD_SPI_DEFAULT_PIN_MISO, \
    .pin_mosi = SD_CARD_SPI_DEFAULT_PIN_MOSI, \
    .pin_clk = SD_CARD_SPI_DEFAULT_PIN_CLK, \
    .pin_cs = SD_CARD_SPI_DEFAULT_PIN_CS, \
    .spi_host = SD_CARD_SPI_DEFAULT_HOST, \
    .spi_freq_khz = SD_CARD_SPI_DEFAULT_FREQ_KHZ \
}

/**
 * @brief Runtime SD card object with mount and unmount operations.
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
 * @brief Initializes the SD card object (defaults to Adafruit SDIO 4-bit mode if config is NULL).
 */
esp_err_t sd_card_init(sd_card_t *sd, const sd_card_config_t *config);

/**
 * @brief Explicitly initialize for Adafruit SDIO 4-bit mode.
 */
esp_err_t sd_card_init_sdio(sd_card_t *sd, const sd_card_sdio_config_t *sdio_cfg);

/**
 * @brief Explicitly initialize for legacy SPI mode.
 */
esp_err_t sd_card_init_spi(sd_card_t *sd, const sd_card_spi_config_t *spi_cfg);

/**
 * @brief Read file content into heap-allocated buffer. Caller must free() returned pointer.
 */
uint8_t* sd_card_read_file(const char* path, size_t* out_size);

/**
 * @brief Save file to SD card.
 */
bool sd_card_save_file(const char *path, const uint8_t *data, size_t size);

/**
 * @brief Delete file from SD card.
 */
bool sd_card_delete_file(const char *path);

#ifdef __cplusplus
}
#endif

#endif // SD_CARD_H