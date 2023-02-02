#!/usr/bin/env bash
set -ex

# First, we assume you have installed vulkan by following
# windows: https://vulkan.lunarg.com/doc/sdk/latest/windows/getting_started.html
# linux: https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started.html
# macOS: https://vulkan.lunarg.com/doc/sdk/latest/mac/getting_started.html
#
# You can read ./install-vulkan-macos.md for a note about installation on macOS.

dir=build-android-arm64-v8a-with-vulkan

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

if [ -z $VULKAN_SDK ]; then
  VULKAN_SDK=/Users/fangjun/software/vulkansdk/1.3.236.0/macOS
fi

if [ ! -d $VULKAN_SDK ]; then
  echo "Please install Vulkan SDK first. Please see ./install-vulkan-macos.md"
  exit 1
fi

echo "ANDROID_NDK: $ANDROID_NDK"
echo "VULKAN_SDK: $VULKAN_SDK"
sleep 1

if [ ! -e my-glslang/build/install/lib/libglslang.so ]; then
  if [ ! -d my-glslang ]; then
    git clone https://github.com/KhronosGroup/glslang.git my-glslang
  fi

  pushd my-glslang
  # Note: the master branch of ncnn is using the following commit
  git checkout 88fd417b0bb7d91755961c70e846d274c182f2b0

  mkdir -p build
  cd build

  if [ $(uname) == "Darwin" ]; then
    os=darwin
  elif [ $(uname) == "Linux" ]; then
    os=linux
  else
    echo "Unsupported system: $(uname -a)"
    exit 1
  fi

  cmake \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX="$(pwd)/install" \
    -DANDROID_ABI=arm64-v8a \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_SYSTEM_NAME=Android \
    -DANDROID_TOOLCHAIN=clang \
    -DANDROID_ARM_MODE=arm \
    -DCMAKE_MAKE_PROGRAM=$ANDROID_NDK/prebuilt/${os}-x86_64/bin/make \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    ..

  make -j4
  make install/strip
  ls -lh install/lib/

  echo "Finish building glslang"
  sleep 1

  popd
fi

cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -Dglslang_DIR=$PWD/my-glslang/build/install/lib/cmake/glslang \
    -DANDROID_USE_LEGACY_TOOLCHAIN_FILE=False \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DNCNN_SYSTEM_GLSLANG=ON \
    -DSHERPA_NCNN_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_NCNN_ENABLE_BINARY=OFF \
    -DSHERPA_NCNN_ENABLE_TEST=OFF \
    -DSHERPA_NCNN_ENABLE_C_API=OFF \
    -DSHERPA_NCNN_ENABLE_GENERATE_INT8_SCALE_TABLE=OFF \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DANDROID_ABI="arm64-v8a" \
    -DNCNN_VULKAN=ON \
    -DANDROID_PLATFORM=android-24 ..

make VERBOSE=1 -j4
make install/strip

cp -v my-glslang/build/install/lib/libSPIRV.so install/lib/
cp -v my-glslang/build/install/lib/libglslang.so install/lib/
