idf_component_register(
    SRCS 
        tf_wrapper.cpp
       
    INCLUDE_DIRS 
        "."
    REQUIRES 
        heap
        esp_system
        espressif__esp-tflite-micro
)

target_compile_options(${COMPONENT_LIB} PRIVATE -std=c++17)
target_compile_definitions(${COMPONENT_LIB} PUBLIC TF_LITE_STATIC_MEMORY)