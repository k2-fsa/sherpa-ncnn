#!/usr/bin/env bash

set -ex

if [ ! -d ../build-swift-macos ]; then
  echo "Please run ../build-swift-macos.sh first!"
  exit 1
fi

if [ ! -d ./sherpa-ncnn-conv-emformer-transducer-2022-12-06 ]; then
  echo "Please download the pre-trained model for testing."
  echo "You can refer to"
  echo ""
  echo "https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/conv-emformer-transducer-models.html#csukuangfj-sherpa-ncnn-conv-emformer-transducer-2022-12-06-chinese-english"
  echo "for help"
  exit 1
fi

if [ ! -e ./decode-file ]; then
  # Note: We use -lc++ to link against libc++ instead of libstdc++
  swiftc \
    -lc++ \
    -I ../build-swift-macos/sherpa-ncnn.framework/Headers/ \
    -import-objc-header ./SherpaNcnn-Bridging-Header.h \
    ./decode-file.swift  ./SherpaNcnn.swift \
    -F ../build-swift-macos/ \
    -framework sherpa-ncnn \
    -framework openmp \
    -o decode-file
else
  echo "./decode-file exists - skip building"
fi

./decode-file
