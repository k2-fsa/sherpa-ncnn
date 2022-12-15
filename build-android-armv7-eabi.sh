#!/usr/bin/env bash
set -x

# see https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android

dir=build-android-armv7-eabi

mkdir -p $dir
cd $dir

if [ -z $ANDROID_NDK ]; then
  ANDROID_NDK=/ceph-fj/fangjun/software/android-sdk/ndk/21.0.6113669
  # or use
  # ANDROID_NDK=/ceph-fj/fangjun/software/android-ndk
  #
  # Inside the $ANDROID_NDK directory, you can find a binary ndk-build
  # and some other files like the file "build/cmake/android.toolchain.cmake"
fi

echo "ANDROID_NDK: $ANDROID_NDK"
sleep 1

cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_NCNN_ENABLE_PORTAUDIO=OFF \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON \
    -DANDROID_PLATFORM=android-21 ..
make VERBOSE=1 -j4
make install/strip
