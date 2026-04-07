#!/bin/bash

set -ex

SOROUCE_DIR=$1
OUTPUT_DIR=$2

echo "Starting SenseVoice minimum generate script"
CMAKE_DEFS="-DCMAKE_BUILD_TYPE=Release -DSENSE_VOICE_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=OFF -DGGML_NATIVE=OFF"
mkdir -p build
cd build
cmake ${SOROUCE_DIR} ${CMAKE_DEFS}
make -j8
ls -l
#mv lib/libggml*.so ${OUTPUT_DIR}/
mv lib/libsense-voice-core.so ${OUTPUT_DIR}/
echo "end SenseVoice minimum generate script"
