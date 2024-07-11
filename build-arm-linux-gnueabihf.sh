#!/usr/bin/env bash

if ! command -v arm-linux-gnueabihf-gcc  &> /dev/null; then
  echo "Please install a toolchain for cross-compiling."
  echo "You can refer to: "
  echo "  https://k2-fsa.github.io/sherpa/ncnn/install/arm-embedded-linux.html"
  echo "for help."
  exit 1
fi

set -ex

dir=build-arm-linux-gnueabihf
mkdir -p $dir
cd $dir

if [ ! -f alsa-lib/src/.libs/libasound.so ]; then
  echo "Start to cross-compile alsa-lib"
  if [ ! -d alsa-lib ]; then
    git clone --depth 1 --branch v1.2.12 https://github.com/alsa-project/alsa-lib
  fi
  pushd alsa-lib
  CC=arm-linux-gnueabihf-gcc ./gitcompile --host=arm-linux-gnueabihf
  popd
  echo "Finish cross-compiling alsa-lib"
fi

export CPLUS_INCLUDE_PATH=$PWD/alsa-lib/include:$CPLUS_INCLUDE_PATH
export SHERPA_NCNN_ALSA_LIB_DIR=$PWD/alsa-lib/src/.libs


cmake \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_NCNN_ENABLE_PYTHON=OFF \
  -DSHERPA_NCNN_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_NCNN_ENABLE_JNI=OFF \
  -DSHERPA_NCNN_ENABLE_BINARY=ON \
  -DSHERPA_NCNN_ENABLE_TEST=OFF \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabihf.toolchain.cmake \
  ..

make VERBOSE=1 -j4
make install/strip

cp -v $SHERPA_NCNN_ALSA_LIB_DIR/libasound.so* ./install/lib/
