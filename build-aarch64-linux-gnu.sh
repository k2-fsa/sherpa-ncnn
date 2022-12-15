#!/usr/bin/env bash

set -x

dir=build-aarch64-linux-gnu
mkdir -p $dir
cd $dir
cmake \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake \
  ..

make VERBOSE=1 -j4
make install/strip
