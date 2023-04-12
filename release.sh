#!/usr/bin/env bash
#
# Copyright (c)  2023  Xiaomi Corporation
#
# Please see the end of this file for what files it will generate

SHERPA_NCNN_VERSION=$(grep "SHERPA_NCNN_VERSION" ./CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)
echo "SHERPA_NCNN_VERSION: ${SHERPA_NCNN_VERSION}"
dst=v${SHERPA_NCNN_VERSION}

if [ -d $dst ]; then
  echo "$dst exists - skipping"
  exit 0
fi

./build-android-x86-64.sh
./build-android-armv7-eabi.sh
./build-android-x86-64.sh
./build-ios.sh

mkdir -p $dst/jniLibs/arm64-v8a
cp -v ./build-android-arm64-v8a/install/lib/*.so $dst/jniLibs/arm64-v8a/

mkdir -p $dst/jniLibs/armeabi-v7a
cp -v ./build-android-armv7-eabi/install/lib/*.so $dst/jniLibs/armeabi-v7a/

mkdir -p $dst/jniLibs/x86_64
cp -v ./build-android-x86-64/install/lib/*.so $dst/jniLibs/x86_64

mkdir -p $dst/build-ios/
cp -av ./build-ios/sherpa-ncnn.xcframework $dst/build-ios/
cp -av ./build-ios/openmp.xcframework $dst/build-ios/

cd $dst

tar cjvf sherpa-ncnn-v${SHERPA_NCNN_VERSION}-pre-compiled-android-libs.tar.bz2 ./jniLibs

tar cjvf sherpa-ncnn-v${SHERPA_NCNN_VERSION}-pre-compiled-ios-libs.tar.bz2 ./build-ios

# .
# ├── build-ios
# │   ├── openmp.xcframework
# │   │   ├── Headers
# │   │   │   └── omp.h
# │   │   ├── Info.plist
# │   │   ├── ios-arm64
# │   │   │   └── libomp.a
# │   │   └── ios-arm64_x86_64-simulator
# │   │       └── libomp.a
# │   └── sherpa-ncnn.xcframework
# │       ├── Headers
# │       │   └── sherpa-ncnn
# │       │       └── c-api
# │       │           └── c-api.h
# │       ├── Info.plist
# │       ├── ios-arm64
# │       │   └── sherpa-ncnn.a
# │       └── ios-arm64_x86_64-simulator
# │           └── sherpa-ncnn.a
# ├── jniLibs
# │   ├── arm64-v8a
# │   │   ├── libkaldi-native-fbank-core.so
# │   │   ├── libncnn.so
# │   │   ├── libsherpa-ncnn-core.so
# │   │   └── libsherpa-ncnn-jni.so
# │   ├── armeabi-v7a
# │   │   ├── libkaldi-native-fbank-core.so
# │   │   ├── libncnn.so
# │   │   ├── libsherpa-ncnn-core.so
# │   │   └── libsherpa-ncnn-jni.so
# │   └── x86_64
# │       ├── libkaldi-native-fbank-core.so
# │       ├── libncnn.so
# │       ├── libsherpa-ncnn-core.so
# │       └── libsherpa-ncnn-jni.so
# ├── sherpa-ncnn-v1.8.1-pre-compiled-android-libs.tar.bz2
# └── sherpa-ncnn-v1.8.1-pre-compiled-ios-libs.tar.bz2
#
# 15 directories, 22 files
