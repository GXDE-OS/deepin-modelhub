cmake_minimum_required(VERSION 3.14)

project(deepin-modelhub)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

## 执行 git submodule update --init 命令
#execute_process(
#    COMMAND git submodule update --init --recursive
#    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#    RESULT_VARIABLE SUBMODULE_UPDATE_RESULT
#)

## 检查子模块更新是否成功
#if(NOT SUBMODULE_UPDATE_RESULT EQUAL "0")
#    message(FATAL_ERROR "git submodule update --init failed: ${SUBMODULE_UPDATE_RESULT}")
#endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

message("Build type: ${CMAKE_BUILD_TYPE}")

OPTION(ENABLE_SENSEVOICE "Build sensevoice.cpp backend." OFF)

add_subdirectory(3rdparty)
add_subdirectory(src)
