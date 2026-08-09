if(CONFIG_MODEL_ARCH_DENSE_ENABLED)
    set(srcs "model_arch_dense.c")
else()
    set(srcs "")
endif()

idf_component_register(
    SRCS ${srcs}
    INCLUDE_DIRS "."
    REQUIRES base_model tf_wrapper heap
)

if(CONFIG_MODEL_ARCH_DENSE_ENABLED)
    # Set compile definitions on all possible tf_wrapper targets
    foreach(target tf_wrapper tflite_wrapper __idf_tf_wrapper)
        if(TARGET ${target})
            target_compile_definitions(${target} PUBLIC
                -DCONFIG_TFL_USE_FULLY_CONNECTED
                -DCONFIG_TFL_USE_RELU
                -DCONFIG_TFL_USE_SOFTMAX
            )
        endif()
    endforeach()
endif()
