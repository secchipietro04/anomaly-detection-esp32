#include "esp_vfs_fat.h"
#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
#include "driver/sdmmc_host.h"
#include "esp_log.h"
#include "esp_err.h"
#include "sdmmc_cmd.h"
#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "sd_card.h"
#include <sys/stat.h>
#include <sys/unistd.h>

static const char *TAG = "SD_CARD";

static esp_err_t sd_mount(sd_card_t *self) {
    if (self == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    if (self->is_mounted) {
        ESP_LOGW(TAG, "SD card already mounted.");
        return ESP_OK;
    }

    esp_vfs_fat_mount_config_t mount_config = {
        .format_if_mount_failed = self->config.format_if_mount_failed,
        .max_files = self->config.max_files,
        .allocation_unit_size = 16 * 1024
    };

    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    host.slot = self->config.spi_host;
    host.max_freq_khz = self->config.spi_freq_khz;

    spi_bus_config_t bus_cfg = {
        .mosi_io_num = self->config.pin_mosi,
        .miso_io_num = self->config.pin_miso,
        .sclk_io_num = self->config.pin_clk,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4000,
    };

    // Initialize SPI bus; treat ESP_ERR_INVALID_STATE as non-fatal (bus already in use)
    esp_err_t ret = spi_bus_initialize(self->config.spi_host, &bus_cfg, SDSPI_DEFAULT_DMA);
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "Failed to initialize SPI bus: %s", esp_err_to_name(ret));
        return ret;
    }

    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = self->config.pin_cs;
    slot_config.host_id = self->config.spi_host;

    ret = esp_vfs_fat_sdspi_mount(self->config.mount_point, &host, &slot_config, &mount_config, &self->card_handle);
    if (ret == ESP_OK) {
        self->is_mounted = true;
        ESP_LOGI(TAG, "SD card mounted successfully at %s (Host: %d, Speed: %d kHz)", 
                 self->config.mount_point, self->config.spi_host, self->config.spi_freq_khz);
    } else {
        ESP_LOGE(TAG, "Failed to mount SD card (%s)", esp_err_to_name(ret));
    }

    return ret;
}

static esp_err_t sd_unmount(sd_card_t *self) {
    if (self == NULL || !self->is_mounted) {
        return ESP_OK;
    }

    // Unmount FAT filesystem and remove SD SPI driver handle
    esp_err_t ret = esp_vfs_fat_sdcard_unmount(self->config.mount_point, self->card_handle);
    if (ret == ESP_OK) {
        self->is_mounted = false;
        self->card_handle = NULL;
        ESP_LOGI(TAG, "SD card unmounted successfully.");
    } else {
        ESP_LOGE(TAG, "Failed to unmount SD card (%s)", esp_err_to_name(ret));
    }

    return ret;
}

esp_err_t sd_card_init(sd_card_t *sd, const sd_card_config_t *config) {
    if (sd == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    sd->card_handle = NULL;
    sd->is_mounted = false;
    sd->mount = sd_mount;
    sd->unmount = sd_unmount;

    if (config != NULL) {
        sd->config = *config;
    } else {
        sd_card_config_t default_cfg = SD_CARD_DEFAULT_CONFIG();
        sd->config = default_cfg;
    }

    return ESP_OK;
}


uint8_t* sd_card_read_file(const char* path, size_t* out_size) {
    *out_size = 0;
    struct stat st;
    if (stat(path, &st) != 0) {
        return NULL;
    }
    
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    
    uint8_t *buf = malloc(st.st_size);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    
    size_t read = fread(buf, 1, st.st_size, f);
    fclose(f);
    
    if (read != st.st_size) {
        free(buf);
        return NULL;
    }
    
    *out_size = st.st_size;
    return buf;
}

bool sd_card_save_file(const char *path, const uint8_t *data, size_t size) {
    // Ensure parent folder exists
    mkdir("/sdcard/models", 0755);
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    size_t written = fwrite(data, 1, size, f);
    fclose(f);
    return (written == size);
}

bool sd_card_delete_file(const char *path) {
    return (unlink(path) == 0);
}