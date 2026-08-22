set(GEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cbor/gen")
set(GENERATED_C_TEL "${GEN_DIR}/telemetry.c")
set(GENERATED_H_TEL "${GEN_DIR}/telemetry.h")
set(SCHEMA_FILE_TEL "${CMAKE_CURRENT_SOURCE_DIR}/../../cbor_schema/telemetry.cddl")

message(STATUS "Generating telemetry cbor files now...")
file(MAKE_DIRECTORY ${GEN_DIR})
execute_process(
    COMMAND zcbor code
            -edc ${SCHEMA_FILE_TEL}
            --output-c ${GENERATED_C_TEL}
            --output-h ${GENERATED_H_TEL}
            --default-max-qty 16
            -t Segment InferencePacket NodeHealthInfo NodeAlert Command NodeCapabilities
    RESULT_VARIABLE ZCBOR_RES
)
if(NOT ZCBOR_RES EQUAL 0)
    message(FATAL_ERROR "Failed to pre-generate CBOR files from telemetry schema.")
endif()


set(GENERATED_C_ENS "${GEN_DIR}/ensemble.c")
set(GENERATED_H_ENS "${GEN_DIR}/ensemble.h")
set(SCHEMA_FILE_ENS "${CMAKE_CURRENT_SOURCE_DIR}/../../cbor_schema/ensemble.cddl")
set(SCHEMA_FILE_MOD "${CMAKE_CURRENT_SOURCE_DIR}/../../cbor_schema/model.cddl")

# Concatenate ensemble and model schemas dynamically at build properties phase
set(SCHEMA_FILE_COMBINED "${CMAKE_CURRENT_BINARY_DIR}/ensemble_combined.cddl")
file(READ "${SCHEMA_FILE_ENS}" CONTENT_ENS)
file(READ "${SCHEMA_FILE_MOD}" CONTENT_MOD)
file(WRITE "${SCHEMA_FILE_COMBINED}" "${CONTENT_ENS}\n${CONTENT_MOD}")

message(STATUS "Generating ensemble cbor files now...")
execute_process(
    COMMAND zcbor code
            -dc ${SCHEMA_FILE_COMBINED}
            --output-c ${GENERATED_C_ENS}
            --output-h ${GENERATED_H_ENS}
            --default-max-qty 16
            -t Payload RouteEntry EnsembleConfig ModelPackage
    RESULT_VARIABLE ZCBOR_RES
)
if(NOT ZCBOR_RES EQUAL 0)
    message(FATAL_ERROR "Failed to pre-generate CBOR files from ensemble/model schema.")
endif()


set(GENERATED_C_DEV "${GEN_DIR}/device.c")
set(GENERATED_H_DEV "${GEN_DIR}/device.h")
set(SCHEMA_FILE_DEV "${CMAKE_CURRENT_SOURCE_DIR}/../../cbor_schema/device.cddl")

message(STATUS "Generating device cbor files now...")
execute_process(
    COMMAND zcbor code
            -dc ${SCHEMA_FILE_DEV}
            --output-c ${GENERATED_C_DEV}
            --output-h ${GENERATED_H_DEV}
            --default-max-qty 16
            -t RuntimeConfig
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
        "quad_buffer/quad_buffer.c"
        "harvester/harvester.c"
        "inferencer/inferencer.c"
        "sd_writer/sd_writer.c"
        "uploader/uploader.c"
        "dump_reader/dump_reader.c"
        "mqtt_handlers/mqtt_handlers.c"
        "drivers/sd_card/sd_card.c"
        "drivers/mems/ism330bx.c"
        "network/mqtt/mqtt.c"
        "network/wifi/wifi.c"
        "optimization/dsp_opt.c"
        ${GEN_SRCS}
    INCLUDE_DIRS
        "."
        "optimization"
        "quad_buffer"
        "harvester"
        "inferencer"
        "sd_writer"
        "uploader"
        "dump_reader"
        "mqtt_handlers"
        "cbor"
        "cbor/gen"
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
        esp-dsp
        nvs_flash
        spi_flash
        fatfs
        sdmmc
        log
        zcbor
        base_model
        model_ensemble
        esp_fft_wrapper
)

nvs_create_partition_image(nvs nvs.csv FLASH_IN_PROJECT)

