#!/usr/bin/env bash
# Copyright (c)  2024  Xiaomi Corporation
#
# This script is to build sherpa-ncnn for WebAssembly (NodeJS)
#
# See also
# https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-webassembly
#
# Please refer to
# https://k2-fsa.github.io/sherpa/ncnn/wasm/index.html
# for more details.

set -ex

if [ x"$EMSCRIPTEN" == x"" ]; then
  if ! command -v emcc &> /dev/null; then
    echo "Please install emscripten first"
    echo ""
    echo "You can use the following commands to install it:"
    echo ""
    echo "git clone https://github.com/emscripten-core/emsdk.git"
    echo "cd emsdk"
    echo "git pull"
    echo "./emsdk install latest"
    echo "./emsdk activate latest"
    echo "source ./emsdk_env.sh"
    exit 1
  else
    EMSCRIPTEN=$(dirname $(realpath $(which emcc)))
  fi
fi

export EMSCRIPTEN=$EMSCRIPTEN
echo "EMSCRIPTEN: $EMSCRIPTEN"
if [ ! -f $EMSCRIPTEN/cmake/Modules/Platform/Emscripten.cmake ]; then
  echo "Cannot find $EMSCRIPTEN/cmake/Modules/Platform/Emscripten.cmake"
  echo "Please make sure you have installed emsdk correctly"
  exit 1
fi

mkdir -p build-wasm-simd-for-nodejs
pushd build-wasm-simd-for-nodejs

export SHERPA_NCNN_IS_USING_BUILD_WASM_SH=ON

cmake \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$EMSCRIPTEN/cmake/Modules/Platform/Emscripten.cmake \
  -DNCNN_THREADS=OFF \
  -DNCNN_OPENMP=OFF \
  -DNCNN_SIMPLEOMP=OFF \
  -DNCNN_RUNTIME_CPU=OFF \
  -DNCNN_SSE2=ON \
  -DNCNN_AVX2=OFF \
  -DNCNN_AVX=OFF \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF \
  \
  -DSHERPA_NCNN_ENABLE_WASM=ON \
  -DSHERPA_NCNN_ENABLE_WASM_FOR_NODEJS=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_NCNN_ENABLE_PYTHON=OFF \
  -DSHERPA_NCNN_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_NCNN_ENABLE_JNI=OFF \
  -DSHERPA_NCNN_ENABLE_BINARY=OFF \
  -DSHERPA_NCNN_ENABLE_TEST=OFF \
  -DSHERPA_NCNN_ENABLE_C_API=ON \
  -DSHERPA_NCNN_ENABLE_GENERATE_INT8_SCALE_TABLE=OFF \
  -DSHERPA_NCNN_ENABLE_FFMPEG_EXAMPLES=OFF \
  ..

make -j2
make install
ls -lh install/bin/wasm
