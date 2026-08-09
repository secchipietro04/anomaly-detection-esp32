set(GEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../main/cbor")
set(GENERATED_C_FT "${GEN_DIR}/datapoints.c")
set(GENERATED_H_FT "${GEN_DIR}/datapoints.h")
set(SCHEMA_FILE_FT "${CMAKE_CURRENT_SOURCE_DIR}/../../cbor_schema/datapoints.cddl")

message(STATUS "Generating datapoints cbor files now...")
file(MAKE_DIRECTORY ${GEN_DIR})
execute_process(
    COMMAND zcbor code
            -ec ${SCHEMA_FILE_FT}
            --output-c ${GENERATED_C_FT}
            --output-h ${GENERATED_H_FT}
            -t Segment DatapointsRT DatapointsFT
    RESULT_VARIABLE ZCBOR_RES
)
if(NOT ZCBOR_RES EQUAL 0)
    message(FATAL_ERROR "Failed to pre-generate CBOR files from datapoints schema.")
endif()


set(GENERATED_C_FT "${GEN_DIR}/ensemble.c")
set(GENERATED_H_FT "${GEN_DIR}/ensemble.h")
set(SCHEMA_FILE_FT "${CMAKE_CURRENT_SOURCE_DIR}/../../cbor_schema/ensemble.cddl")

message(STATUS "Generating ensemble cbor files now...")
file(MAKE_DIRECTORY ${GEN_DIR})
execute_process(
    COMMAND zcbor code
            -dc ${SCHEMA_FILE_FT}
            --output-c ${GENERATED_C_FT}
            --output-h ${GENERATED_H_FT}
            -t EnsebleConfig RouteEntry Model SubModel TensorSlice TensorDataType ModelType
    RESULT_VARIABLE ZCBOR_RES
)
if(NOT ZCBOR_RES EQUAL 0)
    message(FATAL_ERROR "Failed to pre-generate CBOR files from ensemble schema.")
endif()


set(GENERATED_C_DEV "${GEN_DIR}/device.c")
set(GENERATED_H_DEV "${GEN_DIR}/device.h")
set(SCHEMA_FILE_DEV "${CMAKE_CURRENT_SOURCE_DIR}/../../cbor_schema/device.cddl")

message(STATUS "Generating device cbor files now...")
execute_process(
    COMMAND zcbor code
            -edc ${SCHEMA_FILE_DEV}
            --output-c ${GENERATED_C_DEV}
            --output-h ${GENERATED_H_DEV}
            -t DeviceConfig InferenceResult DeviceCommand DeviceStatus SensorRegistration ModelCapability
    RESULT_VARIABLE ZCBOR_RES
)
if(NOT ZCBOR_RES EQUAL 0)
    message(FATAL_ERROR "Failed to pre-generate CBOR files from device schema.")
endif()

file(GLOB GEN_SRCS "${GEN_DIR}/*.c")

idf_component_register(
    SRCS
        "main.cc"
        "main_functions.cc"
        "drivers/sd_card/sd_card.c"
        "drivers/mems/ism330bx.c"
        "network/mqtt/mqtt.c"
        "network/wifi/wifi.c"
        ${GEN_SRCS}
    INCLUDE_DIRS
        "."
        "drivers"
        "drivers/sd_card"
        "drivers/mems"
        "network/mqtt"
        "network/wifi"
    REQUIRES
        driver
        esp_system
        esp_netif
        esp_wifi
        nvs_flash
        spi_flash
        fatfs
        sdmmc
        log
        zcbor
        base_model
        model_arch_va
        model_arch_ca
        model_arch_clstm
        model_arch_dense
        model_arch_fits
        model_arch_tn
        model_arch_mr
        model_arch_stftmcnn
        esp_fft_wrapper
        model_ensemble
        model_arch_generic
)

nvs_create_partition_image(nvs nvs.csv FLASH_IN_PROJECT)

