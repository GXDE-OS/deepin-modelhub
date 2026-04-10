#!/bin/bash

set -ex

SOROUCE_DIR=$1
OUTPUT_DIR=$2

echo "Starting vulkan generate script"
CMAKE_DEFS="-DBUILD_SHARED_LIBS=ON -DGGML_VULKAN=1"
mkdir -p build
cd build
cmake ${SOROUCE_DIR} ${CMAKE_DEFS}
make -j16
mv bin/libllama.so ${OUTPUT_DIR}/libllama-vulkan.so
mv bin/libggml.so ${OUTPUT_DIR}/libggml-vulkan.so
mv bin/libggml-base.so ${OUTPUT_DIR}/libggml-base-vulkan.so
mv bin/libggml-cpu.so ${OUTPUT_DIR}/libggml-cpu-vulkan.so

echo "end vulkan generate script"
