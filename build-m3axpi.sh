#!/usr/bin/env bash

# This script is exclusively for building sherpa-ncnn
# on MAIX-III AXera-Pi
# (https://wiki.sipeed.com/hardware/en/maixIII/ax-pi/axpi.html)
#
# It is not for cross-compiling.
#
# Before running this script, please make sure you have the following
# files:
#
# root@AXERA:~/asr# pwd
# /root/asr
#
# root@AXERA:~/asr# ls -lh
# total 25M
# -rw-r--r--  1  501 staff  59K Feb  2 17:01 kaldi-native-fbank-1.11.tar.gz
# -rw-r--r--  1  501 staff  12M Feb  2 17:01 ncnn-sherpa-0.8.tar.gz
# drwxr-xr-x 15  501 staff 4.0K Feb  2 21:04 sherpa-ncnn-1.4.0
#
#
# Note: It is OK if the versions of the above files are different.
# The two `.tar.gz` files must be placed in $HOME/asr

mkdir -p build
cd build

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_NCNN_ENABLE_PYTHON=OFF \
  -DSHERPA_NCNN_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_NCNN_ENABLE_JNI=OFF \
  -DSHERPA_NCNN_ENABLE_BINARY=ON \
  -DSHERPA_NCNN_ENABLE_TEST=OFF \
  -DSHERPA_NCNN_ENABLE_C_API=OFF \
  -DSHERPA_NCNN_ENABLE_GENERATE_INT8_SCALE_TABLE=OFF \
  ..

make -j 4
cd ..

echo
echo "ls -lh ./build/bin"
# You will find two statically-linked executables in ./build/bin
ls -lh ./build/bin

echo
echo "Please refer to"
echo "https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html"
echo "to download pre-trained models"
