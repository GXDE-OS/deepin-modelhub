#!/bin/bash

set -ex

SOROUCE_DIR=$1
OUTPUT_DIR=$2

echo "Starting avx2 generate script"
CMAKE_DEFS="-DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX512=OFF -DGGML_FMA=ON -DGGML_F16C=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF"
mkdir -p build
cd build
cmake ${SOROUCE_DIR} ${CMAKE_DEFS}
make -j16
mv bin/libllama.so ${OUTPUT_DIR}/libllama-avx2.so
mv bin/libggml.so ${OUTPUT_DIR}/libggml-avx2.so
mv bin/libggml-base.so ${OUTPUT_DIR}/libggml-base-avx2.so
mv bin/libggml-cpu.so ${OUTPUT_DIR}/libggml-cpu-avx2.so

echo "end avx2 generate script"
