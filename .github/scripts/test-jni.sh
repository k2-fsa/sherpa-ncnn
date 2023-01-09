#!/usr/bin/env bash

set -e

mkdir -p build
cd build

cmake \
  -D BUILD_SHARED_LIBS=ON \
  -D SHERPA_NCNN_ENABLE_JNI=ON \
  -D SHERPA_NCNN_ENABLE_PORTAUDIO=OFF \
  -D SHERPA_NCNN_ENABLE_PYTHON=OFF \
  -D SHERPA_NCNN_ENABLE_JNI=ON \
  ..

make -j4
ls -lh lib

cd ..

export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH

cd .github/scripts/

kotlinc-jvm -include-runtime -d main.jar Main.kt WaveReader.kt SherpaNcnn.kt AssetManager.kt

java -Djava.library.path=../../build/lib -jar main.jar
