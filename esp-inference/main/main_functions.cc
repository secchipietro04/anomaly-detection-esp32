#include "main_functions.h"

#include "connection/connection.h"
#include "constants.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "inference/model.h"
#include "inference/rnn_model_handler.h"
#include "output_handler.h"
#include "sensor/sensor_interface.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <math.h>

#define SEQUENCE_LENGTH 40
#define FEATURE_SIZE 3

/*
| esp | mki   |
|-----|-------|
| 9   | sda   |
| 10  | scl   |
| gnd | sdo   |
| gnd | gnd   |
| 3.3 | cs    |
| 3.3 | vdd   |
| 3.3 | vddio |
*/

#define TAG "ISM330BX"

typedef struct {
    char *dataframe_id;
    int *data_to_send;
    size_t data_size;
    size_t current_index;
    RNNModelHandler encoder_handler;
    RNNModelHandler decoder_handler;
} DataFrame;

#define NUM_DATAFRAMES 3       // Number of dataframes
#define DATA_BUFFER_SIZE 513  // Size of the data buffer for each dataframe
#define NUM_SEND_ITERATIONS 10

DataFrame dataframes[NUM_DATAFRAMES];
int send_count[NUM_DATAFRAMES] = {0};  // Track how many times each dataframe is sent
bool are_dataframes_full = false;
bool models_loaded = false;

// Sensor instance
static ism330bx_t sensor;
static sensor_data_t data;

void initialize_dataframes() {
    for (int i = 0; i < NUM_DATAFRAMES; i++) {
        dataframes[i].dataframe_id = NULL;  // Initialize to NULL
        dataframes[i].data_to_send = (int *)malloc(sizeof(int) * DATA_BUFFER_SIZE);
        dataframes[i].data_size = DATA_BUFFER_SIZE;
        dataframes[i].current_index = 0;
    }
}

void free_dataframes() {
    for (int i = 0; i < NUM_DATAFRAMES; i++) {
        if (dataframes[i].data_to_send) {
            free(dataframes[i].data_to_send);
        }
    }
}
int setupSensor() {
    // Initialize I2C driver
    i2c_config_t i2c_config = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master = {.clk_speed = I2C_MASTER_FREQ_HZ},
        .clk_flags = 0,
    };

    esp_err_t ret = i2c_param_config(I2C_MASTER_NUM, &i2c_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure I2C");
        return -1;
    }

    ret = i2c_driver_install(I2C_MASTER_NUM, i2c_config.mode, 0, 0, 0);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to install I2C driver");
        return -1;
    }

    // Create ISM330BX sensor instance
    sensor.i2c_num = I2C_MASTER_NUM;
    sensor.address = ISM330BX_I2C_ADDR;

    // Initialize the sensor
    ret = ism330bx_init(&sensor);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize ISM330BX sensor");
        return -1;
    }

    ESP_LOGI(TAG, "ISM330BX sensor initialized successfully");
    return 0;
}


void setupRNNModels(DataFrame *df) {
    unsigned char *encoder_model = (unsigned char *)malloc(1<<16);
    unsigned char *decoder_model = (unsigned char *)malloc(1<<16);
    if (!encoder_model) {
        ESP_LOGE(TAG, "Failed to allocate memory for encoder model");
        return;
    }
    if (!decoder_model) {
        ESP_LOGE(TAG, "Failed to allocate memory for decoder model");
        free(encoder_model);
        return;
    }
    int encoder_model_size = get_model_dataframe(df->dataframe_id, "encoder", (char *)encoder_model, 1<<16);
    ESP_LOGI(TAG, "Encoder model size: %d", encoder_model_size);
    if (encoder_model_size >= 0) {
        initRNNModelHandler(&df->encoder_handler);
        if (!df->encoder_handler.setup(encoder_model, 1 << 17, 51, 56, 3, 8)) {
            ESP_LOGE(TAG, "Failed to set up encoder for DataFrame %s", df->dataframe_id);
        } else {
            ESP_LOGI(TAG, "Encoder setup complete for DataFrame %s", df->dataframe_id);
        }
    }else{
        ESP_LOGE(TAG, "Failed to get encoder model for DataFrame %s", df->dataframe_id);
    }

    int decoder_model_size = get_model_dataframe(df->dataframe_id, "decoder", (char *)decoder_model, 1<<16);
    if (decoder_model_size >= 0) {
        initRNNModelHandler(&df->decoder_handler);
        if (!df->decoder_handler.setup(decoder_model, 1 << 16, 56, 51, 8, 3)) {
            ESP_LOGE(TAG, "Failed to set up decoder for DataFrame %s", df->dataframe_id);
        } else {
            ESP_LOGI(TAG, "Decoder setup complete for DataFrame %s", df->dataframe_id);
        }
    }else{
        ESP_LOGE(TAG, "Failed to get decoder model for DataFrame %s", df->dataframe_id);
    }

    free(encoder_model);
    free(decoder_model);
}

void processRNNModels(DataFrame *df) {
    float normalized_inputs[SEQUENCE_LENGTH][3];
    float decoder_inputs[8];
    float reconstructed_outputs[SEQUENCE_LENGTH][3];
    float std_dev[3] = {0.0, 0.0, 0.0};

    // Loop to collect and normalize sensor data for SEQUENCE_LENGTH timesteps
    for (size_t i = 0; i < SEQUENCE_LENGTH; i++) {
        int gyro_data[3];  // Array to hold gyro data (x, y, z)

        // Read gyroscope data using the sensor's reading function
        esp_err_t ret = ism330bx_read_gyro(&sensor, &data);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to read gyroscope data at timestep %zu", i);
            return;
        }

        // Normalize the sensor data (assuming the raw range is -32768 to 32767)
        gyro_data[0] = data.gyro_x;
        gyro_data[1] = data.gyro_y;
        gyro_data[2] = data.gyro_z;

        // Normalize the gyro data
        for (size_t j = 0; j < 3; j++) {
            normalized_inputs[i][j] = (float)gyro_data[j] / 32768.0;  // Normalize to [-1, 1] range
        }
    }

    // Check if encoder and decoder handlers are initialized properly
    if (!df->encoder_handler.input || !df->encoder_handler.output) {
        ESP_LOGE(TAG, "Encoder handler not initialized properly for DataFrame %s", df->dataframe_id);
        return;
    }

    if (!df->decoder_handler.input || !df->decoder_handler.output) {
        ESP_LOGE(TAG, "Decoder handler not initialized properly for DataFrame %s", df->dataframe_id);
        return;
    }

    // Feed the data into the encoder for each timestep
    for (size_t t = 0; t < SEQUENCE_LENGTH; t++) {
        // Copy the normalized data for each timestep into the encoder input
        memcpy(df->encoder_handler.input->data.f, normalized_inputs[t], sizeof(float) * 3);
        df->encoder_handler.step();  // Process the encoder with the current input
    }

    // Get the output from the encoder to feed into the decoder
    memcpy(decoder_inputs, df->encoder_handler.output->data.f, sizeof(float) * 8);  // Assuming output is 8-dimensional

    // Feed the decoder with the encoder's output for each timestep
    for (size_t t = 0; t < SEQUENCE_LENGTH; t++) {
        // Copy the decoder input (from encoder output) into the decoder input
        memcpy(df->decoder_handler.input->data.f, decoder_inputs, sizeof(float) * 8);
        df->decoder_handler.step();  // Process the decoder
        memcpy(reconstructed_outputs[t], df->decoder_handler.output->data.f, sizeof(float) * 3);  // Store decoder output
    }

    // Calculate the standard deviation of the difference between input and reconstructed outputs
    for (size_t t = 0; t < SEQUENCE_LENGTH; t++) {
        for (size_t j = 0; j < 3; j++) {
            float diff = normalized_inputs[t][j] - reconstructed_outputs[t][j];
            std_dev[j] += diff * diff;
        }
    }

    // Finalize the standard deviation calculation
    for (size_t j = 0; j < 3; j++) {
        std_dev[j] = sqrt(std_dev[j] / SEQUENCE_LENGTH);
        ESP_LOGI(TAG, "DataFrame %s - Std Dev (dim %zu): %f", df->dataframe_id, j, std_dev[j]);
    }
}

// Function to average gyro data
void read_and_average_gyro(int *output_buffer, size_t merge_factor) {
    int gyro_x = 0, gyro_y = 0, gyro_z = 0;
    for (size_t i = 0; i < merge_factor; i++) {
        esp_err_t ret = ism330bx_read_gyro(&sensor, &data);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to read gyroscope data");
            return;
        }
        gyro_x += data.gyro_x;
        gyro_y += data.gyro_y;
        gyro_z += data.gyro_z;
    }
    output_buffer[0] = gyro_x / merge_factor;
    output_buffer[1] = gyro_y / merge_factor;
    output_buffer[2] = gyro_z / merge_factor;
}

void setup() {
    setupSensor();
    connect_to_wifi();
    initialize_dataframes();
}

void loop() {
    ESP_LOGI(TAG, "Free heap: %lu", esp_get_free_heap_size());
    if (!are_dataframes_full) {
        for (int i = 0; i < NUM_DATAFRAMES; i++) {
            if (!dataframes[i].dataframe_id) {
                dataframes[i].dataframe_id = post_dataframe();
                if (!dataframes[i].dataframe_id) {
                    ESP_LOGE(TAG, "Failed to create dataframe");
                    continue;
                }
                ESP_LOGI(TAG, "Created DataFrame ID: %s", dataframes[i].dataframe_id);
            }
        }
    }

    for (int i = 0; i < NUM_DATAFRAMES; i++) {
        size_t merge_factor = (size_t)pow(2, i);
        while (dataframes[i].current_index < dataframes[i].data_size) {
            int averaged_data[3];
            read_and_average_gyro(averaged_data, merge_factor);
            memcpy(&dataframes[i].data_to_send[dataframes[i].current_index], averaged_data, sizeof(averaged_data));
            dataframes[i].current_index += 3;

            if (dataframes[i].current_index >= dataframes[i].data_size) {
                ESP_LOGI(TAG, "DataFrame %d is full", i);
                break;
            }
        }
    }

    are_dataframes_full = true;
    for (int i = 0; i < NUM_DATAFRAMES; i++) {
        if (dataframes[i].current_index < dataframes[i].data_size) {
            are_dataframes_full = false;
            break;
        }
    }

    if (are_dataframes_full) {
        for (int i = 0; i < NUM_DATAFRAMES; i++) {
            if (send_count[i] < NUM_SEND_ITERATIONS) {
                post_dataframe_id(dataframes[i].dataframe_id, dataframes[i].data_to_send, dataframes[i].data_size);
                dataframes[i].current_index = 0;
                send_count[i]++;
                ESP_LOGI(TAG, "DataFrame %d sent successfully (%d/%d)", i, send_count[i], NUM_SEND_ITERATIONS);
            }
        }

        bool all_sent = true;
        for (int i = 0; i < NUM_DATAFRAMES; i++) {
            if (send_count[i] < NUM_SEND_ITERATIONS) {
                all_sent = false;
                break;
            }
        }

        if (all_sent && !models_loaded) {
            for (int i = 0; i < NUM_DATAFRAMES; i++) {
                setupRNNModels(&dataframes[i]);
            }
            models_loaded = true;
        }

        if (models_loaded) {
            for (int i = 0; i < NUM_DATAFRAMES; i++) {
                processRNNModels(&dataframes[i]);
            }
        }

        are_dataframes_full = false;
    }
}
