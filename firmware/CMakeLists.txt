cmake_minimum_required(VERSION 3.5)
list(APPEND EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/components/ai")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(firmware)
