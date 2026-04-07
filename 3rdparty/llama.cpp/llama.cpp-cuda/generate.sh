#!/bin/bash

set -ex

SOROUCE_DIR=$1
OUTPUT_DIR=$2

echo "Starting cuda generate script"
if [ -z "${CUDACXX}" ]; then
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        export CUDACXX=/usr/local/cuda/bin/nvcc
    else
        # Try the default location in case it exists
        export CUDACXX=$(command -v nvcc)
    fi
fi

#https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=compute_#ptx-compatibility
#or nvcc --help #--gpu-architecture <arch>
CMAKE_DEFS="-DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX512=OFF -DGGML_FMA=ON -DGGML_F16C=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF"
#DCMAKE_CUDA_ARCHITECTURES=all
mkdir -p build
cd build
cmake ${SOROUCE_DIR} ${CMAKE_DEFS}
make -j16
mv bin/libllama.so ${OUTPUT_DIR}/libllama-cuda.so
mv bin/libggml.so ${OUTPUT_DIR}/libggml-cuda.so
mv bin/libggml-base.so ${OUTPUT_DIR}/libggml-base-cuda.so
mv bin/libggml-cpu.so ${OUTPUT_DIR}/libggml-cpu-cuda.so
mv bin/libggml-cuda.so ${OUTPUT_DIR}/libggml-cuda-cuda.so
echo "end cuda generate script"
