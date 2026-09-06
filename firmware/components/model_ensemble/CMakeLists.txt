if(CONFIG_MODEL_ENSELE_ENABLED OR CONFIG_MODEL_ENSEMBLE_ENABLED)
    set(srcs 
        "model_ensemble.c"
        "dsp_utils.c"
        "loss_utils.c"
        "model_cache.c"
    )
else()
    set(srcs "")
endif()

idf_component_register(
    SRCS ${srcs}
    INCLUDE_DIRS "."
    REQUIRES base_model tf_wrapper heap esp_fft_wrapper
)
