if(CONFIG_MODEL_ENSEMBLE_ENABLED)
    set(srcs "model_ensemble.c")
else()
    set(srcs "")
endif()

idf_component_register(
    SRCS ${srcs}
    INCLUDE_DIRS "."
    REQUIRES base_model tf_wrapper heap
)
