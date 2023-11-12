#!/usr/bin/env bash
# Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)

npm list | grep ffi-napi >/dev/null || npm install ffi-napi
npm list | grep ref-struct-napi >/dev/null || npm install ref-struct-napi
npm list | grep wav >/dev/null || npm install wav

if [ ! -e ./install ]; then
  cd ..
  mkdir -p build
  cd build
  cmake -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DSHERPA_NCNN_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_NCNN_ENABLE_BINARY=OFF \
    -DSHERPA_NCNN_ENABLE_C_API=ON \
    -DSHERPA_NCNN_ENABLE_GENERATE_INT8_SCALE_TABLE=OFF \
    -DSHERPA_NCNN_ENABLE_PYTHON=OFF \
    ..
  make -j3
  make install
  cd ../nodejs-examples
  ln -s $PWD/../build/install .
fi

if [ ! -d ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13 ]; then
  echo "Please refer to"
  echo "https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/zipformer-transucer-models.html#csukuangfj-sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13-bilingual-chinese-english"
  echo "to download the models"
  exit 0
fi

node ./test.js
