#ifndef ISM330BX_H
#define ISM330BX_H

#include "driver/i2c.h"
#include "esp_err.h"
#include "esp_log.h"

// I2C Configuration
#define I2C_MASTER_NUM I2C_NUM_0
// #define I2C_MASTER_SCL_IO 42
// #define I2C_MASTER_SDA_IO 41
#define I2C_MASTER_SCL_IO (gpio_num_t)10
#define I2C_MASTER_SDA_IO (gpio_num_t)9

// 370370 Hz is the maximum frequency for the ISM330BX (even with 2.2k pull-up resistors)
// using 350 kHz to be safe
#define I2C_MASTER_FREQ_HZ 350000
#define I2C_MASTER_TX_BUF_DISABLE 0
#define I2C_MASTER_RX_BUF_DISABLE 0

// ISM330BX I2C Address
#define ISM330BX_I2C_ADDR 0x6A  // Change to 0x6B if SDO is high

// ISM330BX Register Definitions
#define WHO_AM_I_REG 0x0F
#define CTRL1_XL_REG 0x10
#define CTRL2_G_REG 0x11
#define OUTX_L_G_REG 0x22
#define OUTX_H_G_REG 0x23
#define OUTY_L_G_REG 0x24
#define OUTY_H_G_REG 0x25
#define OUTZ_L_G_REG 0x26
#define OUTZ_H_G_REG 0x27

#define OUTX_L_A_REG 0x28
#define OUTX_H_A_REG 0x2C
#define OUTY_L_A_REG 0x2A
#define OUTY_H_A_REG 0x2B
#define OUTZ_L_A_REG 0x29
#define OUTZ_H_A_REG 0x2D

#define TIMESTAMP0_REG 0x40
#define TIMESTAMP1_REG 0x41
#define TIMESTAMP2_REG 0x42
#define TIMESTAMP3_REG 0x43


/*
 Register bit layout:
 Bit  |    7 |  6              |  5            |  4            |  3        |  2        |  1        |  0
 Name |    0 | OP_MODE_G_2     | OP_MODE_G_1   | OP_MODE_G_0   | ODR_G_3   | ODR_G_2   | ODR_G_1   | ODR_G_0
 1. Note: Bit 7 (OP_MODE_G_2) must be set to 0 for the correct operation of the
evice.
 Description:
 OP_MODE_G_[2:0] - Gyroscope Operating Mode Selection:
   - 000: High-performance mode (default)
   - 001: Reserved
   - 010: Reserved
   - 011: Reserved
   - 100: Sleep mode
   - 101: Low-power mode
   - 110-111: Reserved
 ODR_G_[3:0] - Gyroscope Output Data Rate Selection:
   - Refer to Table 53 for available ODR options:
     | ODR_G_3 | ODR_G_2 | ODR_G_1 | ODR_G_0 | ODR [Hz]              |
     |---------|---------|---------|---------|-----------------------|
     |   0     |   0     |   0     |   0     | Power-down (default)  |
     |   0     |   0     |   1     |   0     | 7.5 Hz                |
     |   0     |   0     |   1     |   1     | 15 Hz                 |
     |   0     |   1     |   0     |   0     | 30 Hz                 |
     |   0     |   1     |   0     |   1     | 60 Hz                 |
     |   0     |   1     |   1     |   0     | 120 Hz                |
     |   0     |   1     |   1     |   1     | 240 Hz                |
     |   1     |   0     |   0     |   0     | 480 Hz (high-perf)    |
     |   1     |   0     |   0     |   1     | 960 Hz (high-perf)    |
     |   1     |   0     |   1     |   0     | 1.92 kHz (high-perf)  |
     |   1     |   0     |   1     |   1     | 3.84 kHz (high-perf)  |
     |   Others                                | Reserved
*/
#define CONFIG_GYRO         0b00001011
#define CONFIG_ACCELERATION 0b00000000

//enable timestamp
#define FUNCTIONS_ENABLE_TIMESTAMP 0b01000000
#define FUNCTIONS_DISABLE_TIMESTAMP 0b00000000
#define FUNCTIONS_ENABLE_REG 0x50
#define FUNCTIONS_CONF FUNCTIONS_ENABLE_TIMESTAMP

// Data Structure for Sensor Readings
typedef struct {
    int16_t gyro_x;       // Gyroscope X-axis
    int16_t gyro_y;       // Gyroscope Y-axis
    int16_t gyro_z;       // Gyroscope Z-axis
    int16_t acc_x;        // Accelerometer X-axis
    int16_t acc_y;        // Accelerometer Y-axis
    int16_t acc_z;        // Accelerometer Z-axis
    uint32_t timestamp;   // 32-bit Timestamp
} sensor_data_t;

// ISM330BX Sensor Structure
typedef struct {
    i2c_port_t i2c_num;   // I2C port number
    uint8_t address;      // I2C address of the sensor
} ism330bx_t;

// Function Prototypes
esp_err_t ism330bx_i2c_write(ism330bx_t *sensor, uint8_t reg_addr, uint8_t data);
esp_err_t ism330bx_i2c_read(ism330bx_t *sensor, uint8_t reg_addr, uint8_t *data);
esp_err_t ism330bx_init(ism330bx_t *sensor);

esp_err_t ism330bx_read_gyro(ism330bx_t *sensor, sensor_data_t *data);
esp_err_t ism330bx_read_accel(ism330bx_t *sensor, sensor_data_t *data);
esp_err_t ism330bx_read_timestamp(ism330bx_t *sensor, sensor_data_t *data);

#endif // ISM330BX_H
