#include "ism330bx.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include <stdlib.h>
#include <string.h>

static const char *TAG = "ISM330BX_SPI";
#define ISM330BX_MAX_STACK_BUF 64

// base spi read register
static esp_err_t ism330bx_read_reg_spi(ism330bx_spi_dev_t *dev, uint8_t reg_addr, uint8_t *data, size_t len) {
  if (!dev || !dev->spi_handle || !data || len == 0)
    return ESP_ERR_INVALID_ARG;

  size_t buf_len = len + 1;
  uint8_t stack_tx[ISM330BX_MAX_STACK_BUF];
  uint8_t stack_rx[ISM330BX_MAX_STACK_BUF];

  uint8_t *tx_buf = (buf_len <= ISM330BX_MAX_STACK_BUF) ? stack_tx : (uint8_t *)heap_caps_malloc(buf_len, MALLOC_CAP_DMA);
  uint8_t *rx_buf = (buf_len <= ISM330BX_MAX_STACK_BUF) ? stack_rx : (uint8_t *)heap_caps_malloc(buf_len, MALLOC_CAP_DMA);

  if (!tx_buf || !rx_buf) {
    if (tx_buf && tx_buf != stack_tx)
      free(tx_buf);
    if (rx_buf && rx_buf != stack_rx)
      free(rx_buf);
    return ESP_ERR_NO_MEM;
  }

  memset(tx_buf, 0x00, buf_len);
  tx_buf[0] = reg_addr | ISM330BX_SPI_READ_BIT;

  spi_transaction_t t = {
      .length = 8 * buf_len,
      .tx_buffer = tx_buf,
      .rx_buffer = rx_buf,
  };

  esp_err_t ret = spi_device_polling_transmit(dev->spi_handle, &t);
  if (ret == ESP_OK) {
    memcpy(data, &rx_buf[1], len);
  }

  if (tx_buf != stack_tx)
    free(tx_buf);
  if (rx_buf != stack_rx)
    free(rx_buf);

  return ret;
}

// base spi write register
static esp_err_t ism330bx_write_reg_spi(ism330bx_spi_dev_t *dev, uint8_t reg_addr, const uint8_t *data, size_t len) {
  if (!dev || !dev->spi_handle || !data || len == 0)
    return ESP_ERR_INVALID_ARG;

  size_t buf_len = len + 1;
  uint8_t stack_tx[ISM330BX_MAX_STACK_BUF];
  uint8_t *tx_buf = (buf_len <= ISM330BX_MAX_STACK_BUF) ? stack_tx : (uint8_t *)heap_caps_malloc(buf_len, MALLOC_CAP_DMA);

  if (!tx_buf)
    return ESP_ERR_NO_MEM;

  tx_buf[0] = reg_addr & ISM330BX_SPI_WRITE_BIT;
  memcpy(&tx_buf[1], data, len);

  spi_transaction_t t = {
      .length = 8 * buf_len,
      .tx_buffer = tx_buf,
      .rx_buffer = NULL,
  };

  esp_err_t ret = spi_device_polling_transmit(dev->spi_handle, &t);

  if (tx_buf != stack_tx)
    free(tx_buf);

  return ret;
}

static esp_err_t ism330bx_read_temperature(ism330bx_spi_dev_t *dev, float *temp_c) {
  uint8_t raw_temp[2];
  esp_err_t ret = dev->read_reg(dev, ISM330BX_REG_OUT_TEMP_L, raw_temp, 2);
  if (ret == ESP_OK) {
    // data is in 2's complement format, LSB first
    int16_t raw = (int16_t)((raw_temp[1] << 8) | raw_temp[0]);
    *temp_c = ((float)raw / 256.0f) + 25.0f;
  }
  return ret;
}

// returns the raw temperature value as a positive 16-bit integer (0-65535) for faster processing
static esp_err_t ism330bx_read_temperature_raw(ism330bx_spi_dev_t *dev, uint16_t *temp_raw_pos) {
  if (!dev || !temp_raw_pos)
    return ESP_ERR_INVALID_ARG;

  uint8_t raw_temp[2];
  esp_err_t ret = dev->read_reg(dev, ISM330BX_REG_OUT_TEMP_L, raw_temp, 2);
  if (ret == ESP_OK) {
    // data is in 2's complement format, LSB first

    int16_t raw = (int16_t)((raw_temp[1] << 8) | raw_temp[0]);
    *temp_raw_pos = (uint16_t)(raw + 32768);
  }
  return ret;
}

static esp_err_t ism330bx_fetch_fifo_buffer(ism330bx_spi_dev_t *dev, ism330bx_fifo_result_t *result) {
  if (!dev || !result)
    return ESP_ERR_INVALID_ARG;

  uint8_t status[2];
  esp_err_t ret = dev->read_reg(dev, ISM330BX_REG_FIFO_STATUS1, status, 2);
  if (ret != ESP_OK)
    return ret;

  /*
  status[0] (FIFO_STATUS1): Holds the lower 8 bits
  status[1] (FIFO_STATUS2): Holds the upper 2 bits
  status[1] & ISM330BX_FIFO_STAT2_DIFF_MASK selects the upper 2 bits, then shifts left
  and or with status[0] to get the total number of samples.
  */
  uint16_t fifo_samples = status[0] | ((status[1] & ISM330BX_FIFO_STAT2_DIFF_MASK) << 8);
  if (fifo_samples == 0) {
    result->accel_count = 0;
    result->gyro_count = 0;
    return ESP_OK;
  }

  result->accel_count = 0;
  result->gyro_count = 0;

  for (uint16_t i = 0; i < fifo_samples; i++) {
    uint8_t packet[7];
    ret = dev->read_reg(dev, ISM330BX_REG_FIFO_DATA_TAG, packet, 7);
    if (ret != ESP_OK)
      break;

    uint8_t tag = (packet[0] >> ISM330BX_FIFO_TAG_SHIFT) & ISM330BX_FIFO_TAG_MASK;

    if (tag == ISM330BX_FIFO_TAG_GYRO && result->gyro_count < result->gyro_data_size) {
      result->gyro_data[result->gyro_count].x = (int16_t)((packet[2] << 8) | packet[1]);
      result->gyro_data[result->gyro_count].y = (int16_t)((packet[4] << 8) | packet[3]);
      result->gyro_data[result->gyro_count].z = (int16_t)((packet[6] << 8) | packet[5]);
      result->gyro_count++;
    } else if (tag == ISM330BX_FIFO_TAG_ACCEL && result->accel_count < result->accel_data_size) {
      result->accel_data[result->accel_count].x = (int16_t)((packet[2] << 8) | packet[1]);
      result->accel_data[result->accel_count].y = (int16_t)((packet[4] << 8) | packet[3]);
      result->accel_data[result->accel_count].z = (int16_t)((packet[6] << 8) | packet[5]);
      result->accel_count++;
    }
  }
  dev->read_temp(dev, &result->temp_celsius);

  return ESP_OK;
}

static esp_err_t ism330bx_apply_config(ism330bx_spi_dev_t *dev, const ism330bx_config_t *cfg) {
  if (!dev || !cfg)
    return ESP_ERR_INVALID_ARG;

  dev->config = *cfg;

  // configure accel output data rate and full-scale range
  uint8_t ctrl1_xl = (cfg->accel_odr << 4) | (cfg->accel_fs << 2);
  esp_err_t ret = dev->write_reg(dev, ISM330BX_REG_CTRL1_XL, &ctrl1_xl, 1);
  if (ret != ESP_OK)
    return ret;

  // configure gyro output data rate and full-scale range
  uint8_t ctrl2_g = (cfg->gyro_odr << 4) | (cfg->gyro_fs << 2);
  ret = dev->write_reg(dev, ISM330BX_REG_CTRL2_G, &ctrl2_g, 1);
  if (ret != ESP_OK)
    return ret;

  // set fifo decimation for gyro and accel streams
  uint8_t fifo_ctrl3 = ((cfg->fifo_bdr_gy & 0x0F) << 4) | (cfg->fifo_bdr_xl & 0x0F);
  ret = dev->write_reg(dev, ISM330BX_REG_FIFO_CTRL3, &fifo_ctrl3, 1);
  if (ret != ESP_OK)
    return ret;

  // select fifo operating mode
  uint8_t fifo_ctrl4 = cfg->fifo_mode & 0x07;
  ret = dev->write_reg(dev, ISM330BX_REG_FIFO_CTRL4, &fifo_ctrl4, 1);

  return ret;
}

static esp_err_t ism330bx_init(ism330bx_spi_dev_t *dev) {
  uint8_t chip_id = 0;
  esp_err_t ret = dev->read_reg(dev, ISM330BX_REG_WHO_AM_I, &chip_id, 1);
  if (ret != ESP_OK || chip_id != ISM330BX_WHO_AM_I_VAL) {
    ESP_LOGE(TAG, "SPI Init failed: Expected WHO_AM_I 0x71, got 0x%02X", chip_id);
    return ESP_ERR_NOT_FOUND;
  }

  return dev->apply_config(dev, &dev->config);
}

esp_err_t ism330bx_spi_create(ism330bx_spi_dev_t *dev, const ism330bx_init_config_t *init_cfg) {
  if (!dev)
    return ESP_ERR_INVALID_ARG;

  ism330bx_init_config_t cfg = (init_cfg != NULL) ? *init_cfg : (ism330bx_init_config_t)ISM330BX_DEFAULT_INIT_CONFIG();

  // internal spi bus setup
  spi_bus_config_t buscfg = {
      .mosi_io_num = cfg.hw.mosi_pin,
      .miso_io_num = cfg.hw.miso_pin,
      .sclk_io_num = cfg.hw.sclk_pin,
      .quadwp_io_num = -1,
      .quadhd_io_num = -1,
      .max_transfer_sz = 4096,
  };

  dev->host_id = cfg.hw.host_id;
  dev->bus_initialized_by_driver = false;

  esp_err_t ret = spi_bus_initialize(cfg.hw.host_id, &buscfg, SPI_DMA_CH_AUTO);
  if (ret == ESP_OK) {
    dev->bus_initialized_by_driver = true;
  } else if (ret != ESP_ERR_INVALID_STATE) {
    // esp_err_invalid_state means the bus was already initialized by another driver on the same host
    return ret;
  }

  // device attachment
  spi_device_interface_config_t devcfg = {
      .mode = 0,
      .clock_speed_hz = cfg.hw.clock_speed_hz,
      .spics_io_num = cfg.hw.cs_pin,
      .queue_size = 7,
  };

  ret = spi_bus_add_device(cfg.hw.host_id, &devcfg, &dev->spi_handle);
  if (ret != ESP_OK) {
    if (dev->bus_initialized_by_driver)
      spi_bus_free(cfg.hw.host_id);
    return ret;
  }

  // function bindings
  dev->read_reg = ism330bx_read_reg_spi;
  dev->write_reg = ism330bx_write_reg_spi;
  dev->read_temp = ism330bx_read_temperature;
  dev->fetch_fifo_buffer = ism330bx_fetch_fifo_buffer;
  dev->apply_config = ism330bx_apply_config;
  dev->init = ism330bx_init;
  dev->config = cfg.sensor;

  // sensor handshake and config apply
  ret = dev->init(dev);
  if (ret != ESP_OK) {
    spi_bus_remove_device(dev->spi_handle);
    if (dev->bus_initialized_by_driver)
      spi_bus_free(cfg.hw.host_id);
    return ret;
  }

  return ESP_OK;
}

esp_err_t ism330bx_spi_destroy(ism330bx_spi_dev_t *dev) {
  if (!dev || !dev->spi_handle)
    return ESP_ERR_INVALID_ARG;

  esp_err_t ret = spi_bus_remove_device(dev->spi_handle);
  if (dev->bus_initialized_by_driver) {
    spi_bus_free(dev->host_id);
  }
  return ret;
}