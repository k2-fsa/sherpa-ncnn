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
cmake \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabihf.toolchain.cmake \
  ..

make VERBOSE=1 -j4
make install/strip
