if(CONFIG_ESP_FFT_WRAPPER_ENABLED)
    set(srcs "esp_fft_wrapper.c")
else()
    set(srcs "")
endif()

idf_component_register(
    SRCS ${srcs}
    INCLUDE_DIRS "."
    REQUIRES esp-dsp
)
