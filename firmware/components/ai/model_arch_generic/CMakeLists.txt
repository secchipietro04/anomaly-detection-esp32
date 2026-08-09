if(CONFIG_MODEL_ARCH_GENERIC_ENABLED)
    set(srcs "model_arch_generic.c")
else()
    set(srcs "")
endif()

idf_component_register(
    SRCS ${srcs}
    INCLUDE_DIRS "."
    REQUIRES base_model tf_wrapper heap
)
