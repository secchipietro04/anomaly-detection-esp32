file(MAKE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../../../main/cbor")

idf_component_register(
    SRCS "base_model.cpp"
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}" "${CMAKE_CURRENT_LIST_DIR}/../../../main/cbor"
    REQUIRES tf_wrapper heap zcbor
)