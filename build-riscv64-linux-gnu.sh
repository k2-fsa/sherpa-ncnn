#!/usr/bin/env bash

if ! command -v riscv64-linux-gnu-g++  &> /dev/null; then
  echo "Please install the toolchain first."
  echo
  echo "You can use the following command to install the toolchain:"
  echo
  echo "  sudo apt-get install gcc-riscv64-linux-gnu"
  echo "  sudo apt-get install g++-riscv64-linux-gnu"
  echo
  exit 1
fi

dir=build-riscv64-linux-gnu
mkdir -p $dir
cd $dir

if [ ! -f alsa-lib/src/.libs/libasound.so ]; then
  echo "Start to cross-compile alsa-lib"
  if [ ! -d alsa-lib ]; then
    git clone --depth 1 --branch v1.2.12 https://github.com/alsa-project/alsa-lib
  fi
  # If it shows:
  #  ./gitcompile: line 79: libtoolize: command not found
  # Please use:
  #  sudo apt-get install libtool m4 automake
  #
  pushd alsa-lib
  CC=riscv64-linux-gnu-gcc ./gitcompile --host=riscv64-linux-gnu
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
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/riscv64-linux-gnu.toolchain.cmake \
  ..

make VERBOSE=1 -j4
make install/strip

cp -v $SHERPA_NCNN_ALSA_LIB_DIR/libasound.so* ./install/lib/
