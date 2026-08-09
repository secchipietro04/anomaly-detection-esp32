if(CONFIG_MODEL_ARCH_STFTMCNN_ENABLED)
    set(srcs "model_arch_stftmcnn.c")
else()
    set(srcs "")
endif()

idf_component_register(
    SRCS ${srcs}
    INCLUDE_DIRS "."
    REQUIRES base_model tf_wrapper heap esp_fft_wrapper
)

if(CONFIG_MODEL_ARCH_STFTMCNN_ENABLED)
    # Set compile definitions on all possible tf_wrapper targets
    foreach(target tf_wrapper tflite_wrapper __idf_tf_wrapper)
        if(TARGET ${target})
            target_compile_definitions(${target} PUBLIC
                -DCONFIG_TFL_USE_CONV_2D
                -DCONFIG_TFL_USE_FULLY_CONNECTED
                -DCONFIG_TFL_USE_RELU
                -DCONFIG_TFL_USE_SOFTMAX
                -DCONFIG_TFL_USE_MAX_POOL_2D
            )
        endif()
    endforeach()
endif()
