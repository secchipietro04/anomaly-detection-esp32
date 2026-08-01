#ifndef ISM330BX_SPI_H
#define ISM330BX_SPI_H

#include "driver/spi_master.h"
#include "esp_err.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// register addresses
typedef enum {
  ISM330BX_REG_FIFO_CTRL3 = 0x09,
  ISM330BX_REG_FIFO_CTRL4 = 0x0A,
  ISM330BX_REG_WHO_AM_I = 0x0F,
  ISM330BX_REG_CTRL1_XL = 0x10,
  ISM330BX_REG_CTRL2_G = 0x11,
  ISM330BX_REG_OUT_TEMP_L = 0x1D,
  ISM330BX_REG_FIFO_STATUS1 = 0x3A,
  ISM330BX_REG_FIFO_STATUS2 = 0x3B,
  ISM330BX_REG_FIFO_DATA_TAG = 0x78,
} ism330bx_reg_t;

#define ISM330BX_WHO_AM_I_VAL 0x71
#define ISM330BX_SPI_READ_BIT 0x80
#define ISM330BX_SPI_WRITE_BIT 0x7F

// status_reg bitmasks
#define ISM330BX_STAT_XLDA (1 << 0) // bit 0: accelerometer data available
#define ISM330BX_STAT_GDA (1 << 1)  // bit 1: gyroscope data available
#define ISM330BX_STAT_TDA (1 << 2)  // bit 2: temperature data available

// fifo_status2 bitmasks (register 0x3B)
#define ISM330BX_FIFO_STAT2_DIFF_MASK 0x03 // bits 0-1: high bits for fifo sample count
#define ISM330BX_FIFO_STAT2_WTM (1 << 5)   // bit 5: fifo watermark reached
#define ISM330BX_FIFO_STAT2_OVR (1 << 6)   // bit 6: fifo overrun (data loss)
#define ISM330BX_FIFO_STAT2_FULL (1 << 7)  // bit 7: fifo completely full

#define ISM330BX_FIFO_TAG_GYRO 0x01
#define ISM330BX_FIFO_TAG_ACCEL 0x02
#define ISM330BX_FIFO_TAG_SHIFT 3
#define ISM330BX_FIFO_TAG_MASK 0x1F

// configuration enums

typedef enum {
  ISM330BX_ODR_OFF = 0x00,
  ISM330BX_ODR_15Hz = 0x01,
  ISM330BX_ODR_30Hz = 0x02,
  ISM330BX_ODR_60Hz = 0x03,
  ISM330BX_ODR_120Hz = 0x04,
  ISM330BX_ODR_240Hz = 0x05,
  ISM330BX_ODR_480Hz = 0x06,
  ISM330BX_ODR_960Hz = 0x07,
  ISM330BX_ODR_1920Hz = 0x08,
  ISM330BX_ODR_3840Hz = 0x09,
  ISM330BX_ODR_7680Hz = 0x0A,
} ism330bx_odr_t;

typedef enum {
  ISM330BX_ACCEL_FS_2G = 0x00,
  ISM330BX_ACCEL_FS_4G = 0x02,
  ISM330BX_ACCEL_FS_8G = 0x03,
  ISM330BX_ACCEL_FS_16G = 0x01,
} ism330bx_accel_fs_t;

typedef enum {
  ISM330BX_GYRO_FS_250DPS = 0x00,
  ISM330BX_GYRO_FS_500DPS = 0x01,
  ISM330BX_GYRO_FS_1000DPS = 0x02,
  ISM330BX_GYRO_FS_2000DPS = 0x03,
} ism330bx_gyro_fs_t;

typedef enum {
  ISM330BX_FIFO_BYPASS = 0x00,
  ISM330BX_FIFO_MODE = 0x01,
  ISM330BX_FIFO_STREAM = 0x06, // continuous overwrite mode
} ism330bx_fifo_mode_t;

// hardware spi configuration struct

typedef struct {
  spi_host_device_t host_id;
  int mosi_pin;
  int miso_pin;
  int sclk_pin;
  int cs_pin;
  int clock_speed_hz;
} ism330bx_hw_config_t;

// sensor register configuration struct

typedef struct {
  ism330bx_odr_t accel_odr;
  ism330bx_accel_fs_t accel_fs;
  ism330bx_odr_t gyro_odr;
  ism330bx_gyro_fs_t gyro_fs;
  ism330bx_fifo_mode_t fifo_mode;
  uint8_t fifo_bdr_xl; // batch data rate for accel: 0x07 = 960hz
  uint8_t fifo_bdr_gy; // batch data rate for gyro: 0x08 = 1920hz
} ism330bx_config_t;

/**
 * @brief unified master initialization config
 */

typedef struct {
  ism330bx_hw_config_t hw;
  ism330bx_config_t sensor;
} ism330bx_init_config_t;

/**
 * @brief master default configuration with standard spi2 gpios and high-rate
 * fifo streaming
 */
#define ISM330BX_DEFAULT_INIT_CONFIG()               \
  {                                                  \
    .hw =                                            \
        {                                            \
            .host_id = SPI2_HOST,                    \
            .mosi_pin = 23,                          \
            .miso_pin = 19,                          \
            .sclk_pin = 18,                          \
            .cs_pin = 5,                             \
            .clock_speed_hz = 10000000, /* 10 MHz */ \
        },                                           \
    .sensor = {                                      \
      .accel_odr = ISM330BX_ODR_960Hz,               \
      .accel_fs = ISM330BX_ACCEL_FS_4G,              \
      .gyro_odr = ISM330BX_ODR_1920Hz,               \
      .gyro_fs = ISM330BX_GYRO_FS_2000DPS,           \
      .fifo_mode = ISM330BX_FIFO_STREAM,             \
      .fifo_bdr_xl = 0x07,                           \
      .fifo_bdr_gy = 0x08,                           \
    }                                                \
  }

// data containers

typedef struct {
  int16_t x;
  int16_t y;
  int16_t z;
} ism330bx_axis3_t;

typedef struct {
  ism330bx_axis3_t *accel_data;
  size_t accel_data_size; // input buffer size
  size_t accel_count;     // number of samples read

  ism330bx_axis3_t *gyro_data;
  size_t gyro_data_size;
  size_t gyro_count;

  float temp_celsius;
} ism330bx_fifo_result_t;

typedef struct ism330bx_spi_dev_t ism330bx_spi_dev_t;

struct ism330bx_spi_dev_t {
  spi_device_handle_t spi_handle;
  spi_host_device_t host_id;
  ism330bx_config_t config;
  bool bus_initialized_by_driver;

  // object methods
  esp_err_t (*read_reg)(ism330bx_spi_dev_t *dev, uint8_t reg_addr,
                        uint8_t *data, size_t len);
  esp_err_t (*write_reg)(ism330bx_spi_dev_t *dev, uint8_t reg_addr,
                         const uint8_t *data, size_t len);
  esp_err_t (*read_temp)(ism330bx_spi_dev_t *dev, float *temp_c);
  esp_err_t (*fetch_fifo_buffer)(ism330bx_spi_dev_t *dev,
                                 ism330bx_fifo_result_t *result_buffer);
  esp_err_t (*apply_config)(ism330bx_spi_dev_t *dev,
                            const ism330bx_config_t *cfg);
  esp_err_t (*init)(ism330bx_spi_dev_t *dev);
};

/**
 * @brief initializes spi bus, adds spi device, verifies chip id, and applies
 * config. pass null for init_cfg to use default configuration values.
 */
esp_err_t ism330bx_spi_create(ism330bx_spi_dev_t *dev,
                              const ism330bx_init_config_t *init_cfg);
esp_err_t ism330bx_spi_destroy(ism330bx_spi_dev_t *dev);

#ifdef __cplusplus
}
#endif

#endif // ISM330BX_SPI_H