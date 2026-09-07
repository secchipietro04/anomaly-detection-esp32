#include "sd_card.h"
#include "esp_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

    esp_err_t ret = ESP_FAIL;

    if (self->config.bus_type == SD_CARD_BUS_SDIO_4BIT) {
        ESP_LOGI(TAG, "Initializing SD card via Adafruit SDIO 4-Bit Host...");
        // Ensure mount point and options propagate to sdio config
        if (self->config.mount_point) {
            self->config.sdio.mount_point = self->config.mount_point;
        }
        self->config.sdio.max_files = self->config.max_files;
        self->config.sdio.format_if_mount_failed = self->config.format_if_mount_failed;
        if (self->config.allocation_unit_size > 0) {
            self->config.sdio.allocation_unit_size = self->config.allocation_unit_size;
        }

        ret = sd_card_sdio_mount(&self->config.sdio, &self->card_handle);
    } else {
        ESP_LOGI(TAG, "Initializing SD card via legacy SPI Host...");
        // Ensure mount point and options propagate to spi config
        if (self->config.mount_point) {
            self->config.spi.mount_point = self->config.mount_point;
        }
        self->config.spi.max_files = self->config.max_files;
        self->config.spi.format_if_mount_failed = self->config.format_if_mount_failed;
        if (self->config.allocation_unit_size > 0) {
            self->config.spi.allocation_unit_size = self->config.allocation_unit_size;
        }

        ret = sd_card_spi_mount(&self->config.spi, &self->card_handle);
    }

    if (ret == ESP_OK) {
        self->is_mounted = true;
    } else {
        ESP_LOGE(TAG, "Failed to mount SD card (ret=0x%x): %s", ret, esp_err_to_name(ret));
    }

    return ret;
}

static esp_err_t sd_unmount(sd_card_t *self) {
    if (self == NULL || !self->is_mounted) {
        return ESP_OK;
    }

    esp_err_t ret = ESP_OK;
    const char *mp = self->config.mount_point ? self->config.mount_point : SD_CARD_DEFAULT_MOUNT_PT;

    if (self->config.bus_type == SD_CARD_BUS_SDIO_4BIT) {
        ret = sd_card_sdio_unmount(mp, self->card_handle);
    } else {
        ret = sd_card_spi_unmount(mp, self->card_handle);
    }

    if (ret == ESP_OK) {
        self->is_mounted = false;
        self->card_handle = NULL;
        ESP_LOGI(TAG, "SD card unmounted successfully from %s.", mp);
    } else {
        ESP_LOGE(TAG, "Failed to unmount SD card: %s", esp_err_to_name(ret));
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

esp_err_t sd_card_init_sdio(sd_card_t *sd, const sd_card_sdio_config_t *sdio_cfg) {
    if (sd == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    sd_card_config_t cfg = SD_CARD_DEFAULT_CONFIG();
    cfg.bus_type = SD_CARD_BUS_SDIO_4BIT;
    if (sdio_cfg != NULL) {
        cfg.sdio = *sdio_cfg;
        cfg.mount_point = sdio_cfg->mount_point;
    }
    return sd_card_init(sd, &cfg);
}

esp_err_t sd_card_init_spi(sd_card_t *sd, const sd_card_spi_config_t *spi_cfg) {
    if (sd == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    sd_card_config_t cfg = SD_CARD_DEFAULT_CONFIG();
    cfg.bus_type = SD_CARD_BUS_SPI;
    if (spi_cfg != NULL) {
        cfg.spi = *spi_cfg;
        cfg.mount_point = spi_cfg->mount_point;
        cfg.pin_miso = spi_cfg->pin_miso;
        cfg.pin_mosi = spi_cfg->pin_mosi;
        cfg.pin_clk  = spi_cfg->pin_clk;
        cfg.pin_cs   = spi_cfg->pin_cs;
        cfg.spi_host = spi_cfg->spi_host;
        cfg.spi_freq_khz = spi_cfg->spi_freq_khz;
    }
    return sd_card_init(sd, &cfg);
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
    
    size_t read_bytes = fread(buf, 1, st.st_size, f);
    fclose(f);
    
    if (read_bytes != st.st_size) {
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