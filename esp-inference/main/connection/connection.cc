#include "connection.h"

#include "../constants.h"
#include "esp_err.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_wifi.h"

void connect_to_wifi() {
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    esp_netif_t *sta_netif = esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    wifi_config_t wifi_config = {
        .sta =
            {
                .ssid = WIFI_SSID,
                .password = WIFI_PASS,
            },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Connecting to Wi-Fi.. .");
    ESP_ERROR_CHECK(esp_wifi_connect());
}
#define MAX_HTTP_OUTPUT_BUFFER 4096

void http_rest_with_url() {
    char *output_buffer =
        (char *)malloc(MAX_HTTP_OUTPUT_BUFFER);  // Buffer to store response
    if (!output_buffer) {
        ESP_LOGE(TAG, "Failed to allocate HTTP response buffer");
        return;
    }
    memset(output_buffer, 0, MAX_HTTP_OUTPUT_BUFFER);
    int content_length = 0;
    int track_length = 0;
    int max_buff = 2048;
    esp_http_client_config_t config = {
        .url = "http://" SERVER_IP ":" STR(SERVER_PORT) "/dataframe",  // URL for GET request
        .timeout_ms = 5000,  // Timeout value in milliseconds
    };
    esp_http_client_handle_t client = esp_http_client_init(&config);
    ESP_LOGI(TAG, "HTTP GET request to: %s", config.url);
    esp_http_client_set_method(client, HTTP_METHOD_GET);
    esp_err_t err = esp_http_client_open(client, 0);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open HTTP connection: %s",
                 esp_err_to_name(err));
    } else {
        content_length = esp_http_client_fetch_headers(client);
        track_length = content_length;
        if (content_length < 0) {
            ESP_LOGE(TAG, "HTTP client fetch headers failed");
        } else {
            // adding new code
            do {
                int data_read = esp_http_client_read_response(
                    client, output_buffer, max_buff);
                if (data_read >= 0) {
                    ESP_LOGI(TAG, "HTTP GET Status = %d, content_length = %llu",
                             esp_http_client_get_status_code(client),
                             esp_http_client_get_content_length(client));
                    // ESP_LOG_BUFFER_CHAR(TAG, output_buffer,
                    // strlen(output_buffer));
                    track_length -= data_read;
                    if (max_buff > track_length) {
                        max_buff = track_length;
                    }
                    ESP_LOGI("http_rest_with_url", "buffer: %s", output_buffer);
                    // ESP_LOGI(TAG, "max_buff = %d, track_length = %d \n",
                    // max_buff, track_length);
                } else {
                    ESP_LOGE(TAG, "Failed to read response");
                }
            } while (track_length > 0);
        }
    }
    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    free(output_buffer);
}
char *post_dataframe() {
    char *output_buffer =
        (char *)malloc(MAX_HTTP_OUTPUT_BUFFER);  // Buffer to store response
    if (!output_buffer) {
        ESP_LOGE(TAG, "Failed to allocate HTTP response buffer");
        return NULL;
    }
    memset(output_buffer, 0, MAX_HTTP_OUTPUT_BUFFER);
    int content_length = 0;
    int track_length = 0;
    int max_buff = 2048;

    char url[128];
    snprintf(url, sizeof(url), "%s/dataframe", base_url);  // Construct URL

    char post_data[] = "\n";
    esp_http_client_config_t config = {
        .url = url,
        .timeout_ms = 5000,  // Timeout in milliseconds
    };

    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (!client) {
        ESP_LOGE(TAG, "Failed to initialize HTTP client");
        free(output_buffer);
        return NULL;
    }

    // Set HTTP method to POST
    esp_http_client_set_method(client, HTTP_METHOD_POST);

    // Set POST data
    esp_http_client_set_post_field(client, post_data, strlen(post_data));

    esp_err_t err = esp_http_client_open(client, strlen(post_data));
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open HTTP connection: %s",
                 esp_err_to_name(err));
        esp_http_client_cleanup(client);
        free(output_buffer);
        return NULL;
    } else {
        content_length = esp_http_client_fetch_headers(client);
        track_length = content_length;

        if (content_length < 0) {
            ESP_LOGE(TAG, "HTTP client fetch headers failed");
        } else {
            char *response = (char *)malloc(
                content_length + 1);  // Allocate memory for the response
            if (!response) {
                ESP_LOGE(TAG, "Failed to allocate memory for response buffer");
                esp_http_client_close(client);
                esp_http_client_cleanup(client);
                free(output_buffer);
                return NULL;
            }

            int total_read = 0;
            do {
                int data_read =
                    esp_http_client_read(client, output_buffer, max_buff);
                if (data_read >= 0) {
                    memcpy(response + total_read, output_buffer, data_read);
                    total_read += data_read;
                    track_length -= data_read;

                    if (max_buff > track_length) {
                        max_buff = track_length;
                    }
                } else {
                    ESP_LOGE(TAG, "Failed to read response");
                    free(response);
                    esp_http_client_close(client);
                    esp_http_client_cleanup(client);
                    free(output_buffer);
                    return NULL;
                }
            } while (track_length > 0);

            response[total_read] = '\0';  // Null-terminate the response
            ESP_LOGI(TAG, "POST Response: %s", response);

            esp_http_client_close(client);
            esp_http_client_cleanup(client);
            free(output_buffer);

            return response;  // Caller is responsible for freeing this memory
        }
    }

    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    free(output_buffer);
    return NULL;
}

int post_dataframe_id(const char *id, int *array, size_t len) {
    char *output_buffer =
        (char *)malloc(MAX_HTTP_OUTPUT_BUFFER);  // Buffer to store response
    if (!output_buffer) {
        ESP_LOGE(TAG, "Failed to allocate HTTP response buffer");
        return -1;
    }
    memset(output_buffer, 0, MAX_HTTP_OUTPUT_BUFFER);
    int content_length = 0;
    int track_length = 0;
    int max_buff = 128;  // read buffer

    // Build the URL
    char url[128];
    snprintf(url, sizeof(url), "%s/dataframe/%s", base_url, id);

    // Initialize HTTP client configuration
    esp_http_client_config_t config = {
        .url = url,          // URL with dataframe ID
        .timeout_ms = 5000,  // Timeout in milliseconds
    };

    esp_http_client_handle_t client = esp_http_client_init(&config);
    esp_http_client_set_method(client,
                               HTTP_METHOD_POST);  // Set HTTP method to POST

    // Build the payload from the array
    char *payload = (char *)malloc(DATAFRAME_PAYLOAD_LENGTH);
    if (!payload) {
        ESP_LOGE(TAG, "Failed to allocate payload buffer");
        esp_http_client_cleanup(client);
        free(output_buffer);
        return -1;
    }
    memset(payload, 0, DATAFRAME_PAYLOAD_LENGTH);
    size_t pos = 0;

    for (size_t i = 0; i < len; i += 3) {
        pos += snprintf(payload + pos, DATAFRAME_PAYLOAD_LENGTH - pos, "%d,%d,%d\n",
                        array[i], array[i + 1], array[i + 2]);

        if (pos >= DATAFRAME_PAYLOAD_LENGTH) {
            ESP_LOGE(TAG, "Payload buffer overflow");
            esp_http_client_cleanup(client);
            free(payload);
            free(output_buffer);
            return -1;
        }
    }

    // ESP_LOGI(TAG, "Payload to send:\n%s", payload);

    // Set POST field and open HTTP connection
    esp_http_client_set_post_field(client, payload, strlen(payload));
    esp_err_t err = esp_http_client_open(client, strlen(payload));
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open HTTP connection: %s",
                 esp_err_to_name(err));
        esp_http_client_close(client);

        esp_http_client_cleanup(client);
        free(payload);
        free(output_buffer);
        return -1;
    }
    int length_written =
        esp_http_client_write(client, payload, strlen(payload));
    if (length_written == ESP_FAIL) {
        ESP_LOGE(TAG, "Failed to write data to server: %s",
                 esp_err_to_name(err));
        esp_http_client_close(client);
        esp_http_client_cleanup(client);
        free(payload);
        free(output_buffer);
        return -1;
    }
    // Fetch response headers and process server response
    content_length = esp_http_client_fetch_headers(client);
    track_length = content_length;

    if (content_length < 0) {
        ESP_LOGE(TAG, "HTTP client fetch headers failed");
        esp_http_client_close(client);
        esp_http_client_cleanup(client);
        free(payload);
        free(output_buffer);
        return -1;
    }

    // Read and process the server response
    int response_number = -1;  // Default to error
    do {
        int data_read = esp_http_client_read(client, output_buffer, max_buff);
        if (data_read >= 0) {
            ESP_LOGI(TAG, "HTTP POST Status = %d, content_length = %llu",
                     esp_http_client_get_status_code(client),
                     esp_http_client_get_content_length(client));
            ESP_LOGI(TAG, "Response buffer: %s", output_buffer);

            // Parse server response to extract the returned number
            if (response_number == -1) {
                response_number =
                    atoi(output_buffer);  // Convert response to integer
            }

            track_length -= data_read;
            if (max_buff > track_length) {
                max_buff = track_length;
            }
        } else {
            ESP_LOGE(TAG, "Failed to read server response");
            esp_http_client_close(client);
            esp_http_client_cleanup(client);
            free(payload);
            free(output_buffer);
            return -1;
        }
    } while (track_length > 0);

    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    free(payload);
    free(output_buffer);

    return response_number;  // Return the parsed server response
}

int get_model_dataframe(const char *id, const char *end_dec, char *buffer, size_t len) {
    char *output_buffer = (char *)malloc(MAX_HTTP_OUTPUT_BUFFER);  // Buffer for HTTP response
    if (!output_buffer) {
        ESP_LOGE(TAG, "Failed to allocate HTTP response buffer");
        return -1;
    }
    memset(output_buffer, 0, MAX_HTTP_OUTPUT_BUFFER);
    int content_length = 0;
    int data_read = -1;

    // Build the URL
    char url[128];
    snprintf(url, sizeof(url), "%s/model/dataframe/%s/%s", base_url, id, end_dec);

    // Initialize HTTP client configuration
    esp_http_client_config_t config = {
        .url = url,          // URL with ID and endpoint
        .timeout_ms = 50000  // Timeout in milliseconds
    };

    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (!client) {
        ESP_LOGE(TAG, "Failed to initialize HTTP client");
        free(output_buffer);
        return -1;
    }
    esp_http_client_set_method(client, HTTP_METHOD_GET);  // Set HTTP method to GET

    // Open the HTTP connection
    esp_err_t err = esp_http_client_open(client, 0);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open HTTP connection: %s", esp_err_to_name(err));
        esp_http_client_close(client);
        esp_http_client_cleanup(client);
        free(output_buffer);
        return -1;
    }

    // Fetch headers to determine content length
    content_length = esp_http_client_fetch_headers(client);
    if (content_length < 0) {
        ESP_LOGE(TAG, "HTTP client fetch headers failed");
        esp_http_client_close(client);
        esp_http_client_cleanup(client);
        free(output_buffer);
        return -1;
    }
    ESP_LOGI(TAG, "Content length: %d, buffer size: %zu", content_length, len);

    if (content_length == -1) {
        ESP_LOGI(TAG, "Chunked transfer encoding detected");
    } else if (content_length >= len) {
        ESP_LOGE(TAG, "Response content length exceeds buffer size");
        esp_http_client_close(client);
        esp_http_client_cleanup(client);
        return -1;
    }

    // Read server response
    int total_data_read = 0;
    do {
        int bytes_read = esp_http_client_read(client, output_buffer, sizeof(output_buffer));
        if (bytes_read > 0) {
            ESP_LOGI(TAG, "Bytes read: %d", bytes_read);

            // Ensure we do not overflow the provided buffer
            if ((total_data_read + bytes_read) >= len) {
                ESP_LOGE(TAG, "Buffer overflow risk, aborting");
                esp_http_client_close(client);
                esp_http_client_cleanup(client);
                free(output_buffer);
                return -1;
            }

            memcpy(buffer + total_data_read, output_buffer, bytes_read);
            total_data_read += bytes_read;
        } else if (bytes_read < 0) {
            ESP_LOGE(TAG, "Failed to read server response");
            esp_http_client_close(client);
            esp_http_client_cleanup(client);
            free(output_buffer);
            return -1;
        }
    } while (content_length == -1 || total_data_read < content_length);

    // Null-terminate the buffer and log the response
    buffer[total_data_read] = '\0';
    ESP_LOGI(TAG, "Dataframe %s response: %s", id, buffer);

    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    free(output_buffer);

    return total_data_read;  // Return the total number of bytes read
}