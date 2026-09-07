#include "sd_card_sdio.h"
#include "esp_vfs_fat.h"
#include "driver/sdmmc_host.h"
#include "driver/gpio.h"
#include "esp_log.h"

static const char *TAG = "SD_CARD_SDIO";

esp_err_t sd_card_sdio_mount(const sd_card_sdio_config_t *config, sdmmc_card_t **out_card) {
    if (!config || !out_card) {
        return ESP_ERR_INVALID_ARG;
    }

    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = config->format_if_mount_failed,
        .max_files = config->max_files,
        .allocation_unit_size = config->allocation_unit_size ? config->allocation_unit_size : 64 * 1024
    };

    // SDMMC Host configuration
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = config->freq_khz ? config->freq_khz : SDMMC_FREQ_HIGHSPEED;

    // Slot configuration for ESP32-S3 custom GPIO matrix
    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
    slot_config.width = 4;
    slot_config.clk = config->pin_clk;
    slot_config.cmd = config->pin_cmd;
    slot_config.d0  = config->pin_d0;
    slot_config.d1  = config->pin_d1;
    slot_config.d2  = config->pin_d2;
    slot_config.d3  = config->pin_d3;
    slot_config.cd  = config->pin_cd;
    slot_config.wp  = SDMMC_SLOT_NO_WP;

    // Enable internal pull-ups on data/cmd lines
    slot_config.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    ESP_LOGI(TAG, "Mounting SD card via SDIO 4-bit (CLK=%d, CMD=%d, D0=%d, D1=%d, D2=%d, D3=%d) @ %d kHz...",
             config->pin_clk, config->pin_cmd, config->pin_d0, config->pin_d1, config->pin_d2, config->pin_d3,
             host.max_freq_khz);

    esp_err_t ret = esp_vfs_fat_sdmmc_mount(config->mount_point, &host, &slot_config, &mount_config, out_card);

    // If 40 MHz fails, try automatic fallback to 20 MHz default speed
    if (ret != ESP_OK && host.max_freq_khz > SDMMC_FREQ_DEFAULT) {
        ESP_LOGW(TAG, "Mount at %d kHz failed (%s). Retrying at %d kHz (Default Speed)...",
                 host.max_freq_khz, esp_err_to_name(ret), SDMMC_FREQ_DEFAULT);
        host.max_freq_khz = SDMMC_FREQ_DEFAULT;
        ret = esp_vfs_fat_sdmmc_mount(config->mount_point, &host, &slot_config, &mount_config, out_card);
    }

    if (ret == ESP_OK) {
        sdmmc_card_t *card = *out_card;
        ESP_LOGI(TAG, "Adafruit SDIO 4-bit mounted at %s | Real Freq: %d kHz | Bus: %d-bit | Sector: %d B",
                 config->mount_point, card->real_freq_khz, (1 << card->log_bus_width), card->csd.sector_size);
    } else {
        ESP_LOGE(TAG, "Failed to mount SD card via SDIO: %s (0x%x)", esp_err_to_name(ret), ret);
    }

    return ret;
}

esp_err_t sd_card_sdio_unmount(const char *mount_point, sdmmc_card_t *card) {
    if (!mount_point || !card) {
        return ESP_OK;
    }
    return esp_vfs_fat_sdcard_unmount(mount_point, card);
}
