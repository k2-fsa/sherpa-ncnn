#!/usr/bin/env bash
set -e

# see https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android

dir=build-android-arm64-v8a

mkdir -p $dir
cd $dir

if [ -z $ANDROID_NDK ]; then
  ANDROID_NDK=/ceph-fj/fangjun/software/android-sdk/ndk/21.0.6113669
  # or use
  # ANDROID_NDK=/ceph-fj/fangjun/software/android-ndk
  #
  # Inside the $ANDROID_NDK directory, you can find a binary ndk-build
  # and some other files like the file "build/cmake/android.toolchain.cmake"

  if [ ! -d $ANDROID_NDK ]; then
    # For macOS, I have installed Android Studio, select the menu
    # Tools -> SDK manager -> Android SDK
    # and set "Android SDK location" to /Users/fangjun/software/my-android
    ANDROID_NDK=/Users/fangjun/software/my-android/ndk/22.1.7171670
  fi
fi

if [ ! -d $ANDROID_NDK ]; then
  echo Please set the environment variable ANDROID_NDK before you run this script
  exit 1
fi

echo "ANDROID_NDK: $ANDROID_NDK"
sleep 1

cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_NCNN_ENABLE_PORTAUDIO=OFF \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-21 ..
make VERBOSE=1 -j4
make install/strip
