

if(NOT CMAKE_BUILD_EARLY_EXPANSION)

    include(FetchContent)

    FetchContent_Declare(
        zcbor_repo
        GIT_REPOSITORY https://github.com/nordicsemi/zcbor.git
        GIT_TAG 0.9.1
    )

    FetchContent_GetProperties(zcbor_repo)
    if(NOT zcbor_repo_POPULATED)
        FetchContent_Populate(zcbor_repo)
    endif()

    idf_component_register(
        SRCS
        "${zcbor_repo_SOURCE_DIR}/src/zcbor_encode.c"
        "${zcbor_repo_SOURCE_DIR}/src/zcbor_decode.c"
        "${zcbor_repo_SOURCE_DIR}/src/zcbor_common.c"
        INCLUDE_DIRS
        "${zcbor_repo_SOURCE_DIR}/include"
    )
    target_compile_options(${COMPONENT_LIB} PRIVATE
        -Wno-strict-aliasing
        -Wno-error=uninitialized
        -fno-strict-aliasing
    )

else()
    idf_component_register()
endif()
