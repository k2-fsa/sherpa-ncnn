#!/usr/bin/env bash

if ! command -v aarch64-linux-gnu-gcc  &> /dev/null; then
  echo "Please install a toolchain for cross-compiling."
  echo "You can refer to: "
  echo "  https://k2-fsa.github.io/sherpa/ncnn/install/aarch64-embedded-linux.html"
  echo "for help."
  exit 1
fi

set -x

dir=build-aarch64-linux-gnu
mkdir -p $dir
cd $dir

if [ ! -f alsa-lib/src/.libs/libasound.so ]; then
  echo "Start to cross-compile alsa-lib"
  if [ ! -d alsa-lib ]; then
    git clone --depth 1 https://github.com/alsa-project/alsa-lib
  fi
  pushd alsa-lib
  CC=aarch64-linux-gnu-gcc ./gitcompile --host=aarch64-linux-gnu
  popd
  echo "Finish cross-compiling alsa-lib"
fi

export CPLUS_INCLUDE_PATH=$PWD/alsa-lib/include:$CPLUS_INCLUDE_PATH
export SHERPA_NCNN_ALSA_LIB_DIR=$PWD/alsa-lib/src/.libs

cmake \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake \
  ..
cp -v $SHERPA_NCNN_ALSA_LIB_DIR/libasound.so* ./install/lib/

make VERBOSE=1 -j4
make install/strip

