#!/usr/bin/env bash
set -e

dir=build-android-arm64-v8a

mkdir -p $dir
cd $dir

# Note from https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android
# (optional) remove the hardcoded debug flag in Android NDK android-ndk
# issue: https://github.com/android/ndk/issues/243
#
# open $ANDROID_NDK/build/cmake/android.toolchain.cmake for ndk < r23
# or $ANDROID_NDK/build/cmake/android-legacy.toolchain.cmake for ndk >= r23
#
# delete "-g" line
#
# list(APPEND ANDROID_COMPILER_FLAGS
#   -g
#   -DANDROID


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
    -DSHERPA_NCNN_ENABLE_BINARY=OFF \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-21 ..
make VERBOSE=1 -j4
make install/strip

# Note: If you set SHERPA_NCNN_ENABLE_BINARY=ON, it will
# generate a binary in $dir/install/bin, which can be run on Android
# with the following commands:
#
# 1. Copy sherpa-ncnn and its dependencies to Android
#
#   cd build-android-arm64-v8a/install/lib
#   adb push ./lib*.so /data/local/tmp
#   cd ../bin
#   adb push ./sherpa-ncnn /data/local/tmp
#
# 2. Login into Android
#
#   adb shell
#   cd /data/local/tmp
#   ./sherpa-ncnn
#
# which shows the following error log:
#
#  CANNOT LINK EXECUTABLE "./sherpa-ncnn": library "libsherpa-ncnn-core.so" not found: needed by main executable
#
# Please run:
#
#  export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
#
# and then you can run:
#
#  ./sherpa-ncnn
#
# It should show the help message of sherpa-ncnn.
