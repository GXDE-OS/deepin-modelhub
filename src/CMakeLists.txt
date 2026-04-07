cmake_minimum_required(VERSION 3.14)

include_directories("${CMAKE_SOURCE_DIR}/include")

set(LIB_PLATFORM_DIR "${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
add_definitions(-DLIB_PLATFORM_DIR="${LIB_PLATFORM_DIR}")

set(EXE_NAME deepin-modelhub)
add_definitions(-DEXE_NAME="${EXE_NAME}")

set(PLUGIN_BACKEND_DIR "${LIB_PLATFORM_DIR}/${EXE_NAME}/backends")

if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DPLUGIN_BACKEND_DIR="${CMAKE_BINARY_DIR}/src")
else()
    add_definitions(-DPLUGIN_BACKEND_DIR="${PLUGIN_BACKEND_DIR}")
endif()

message("LIB_PLATFORM_DIR: ${LIB_PLATFORM_DIR} PLUGIN_BACKEND_DIR ${PLUGIN_BACKEND_DIR}")

if (NOT DEFINED VERSION)
    set(VERSION 0.0.1)
endif()

configure_file(${CMAKE_SOURCE_DIR}/src/environments.h.in ${CMAKE_SOURCE_DIR}/src/environments.h @ONLY)

add_subdirectory(libmodelhub)
add_subdirectory(modelhub)
add_subdirectory(inference-backend)
