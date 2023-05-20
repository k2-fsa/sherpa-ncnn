#!/usr/bin/env bash

export ANDROID_NDK=$ANDROID_NDK_LATEST_HOME

./build-android-arm64-v8a.sh

ls -lh ./build-android-arm64-v8a/install/lib/*.so

cp -v ./build-android-arm64-v8a/install/lib/*.so ./android/SherpaNcnn/app/src/main/jniLibs/arm64-v8a/

echo $PWD

pushd ./android/SherpaNcnn
./gradlew build
echo $PWD
find . -name "*.apk"

