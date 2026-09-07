#include "sd_card_spi.h"
#include "esp_vfs_fat.h"
#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
#include "esp_log.h"

static const char *TAG = "SD_CARD_SPI";

esp_err_t sd_card_spi_mount(const sd_card_spi_config_t *config, sdmmc_card_t **out_card) {
    if (!config || !out_card) {
        return ESP_ERR_INVALID_ARG;
    }

    esp_vfs_fat_mount_config_t mount_config = {
        .format_if_mount_failed = config->format_if_mount_failed,
        .max_files = config->max_files,
        .allocation_unit_size = config->allocation_unit_size ? config->allocation_unit_size : 16 * 1024
    };

    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    host.slot = config->spi_host;
    host.max_freq_khz = config->spi_freq_khz;

    spi_bus_config_t bus_cfg = {
        .mosi_io_num = config->pin_mosi,
        .miso_io_num = config->pin_miso,
        .sclk_io_num = config->pin_clk,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4000,
    };

    esp_err_t ret = spi_bus_initialize(config->spi_host, &bus_cfg, SDSPI_DEFAULT_DMA);
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "Failed to initialize SPI bus: %s", esp_err_to_name(ret));
        return ret;
    }

    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = config->pin_cs;
    slot_config.host_id = config->spi_host;

    ret = esp_vfs_fat_sdspi_mount(config->mount_point, &host, &slot_config, &mount_config, out_card);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Legacy SPI SD card mounted at %s (Host: %d, Speed: %d kHz)", 
                 config->mount_point, config->spi_host, config->spi_freq_khz);
    } else {
        ESP_LOGE(TAG, "Failed to mount SD card via SPI: %s", esp_err_to_name(ret));
    }

    return ret;
}

esp_err_t sd_card_spi_unmount(const char *mount_point, sdmmc_card_t *card) {
    if (!mount_point || !card) {
        return ESP_OK;
    }
    return esp_vfs_fat_sdcard_unmount(mount_point, card);
}
