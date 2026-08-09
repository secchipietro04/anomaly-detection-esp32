# Ensure the generated CBOR directory exists at configure time before registering the component.
# This prevents ESP-IDF's directory validation from failing before the files are generated.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../../../main/cbor")

idf_component_register(
    SRCS "base_model.c"
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}" "${CMAKE_CURRENT_LIST_DIR}/../../../main/cbor"
    REQUIRES tf_wrapper heap zcbor
)