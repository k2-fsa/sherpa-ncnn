#!/usr/bin/env bash

set -ex

if [[ ! -f ../build/lib/libsherpa-ncnn-jni.dylib  && ! -f ../build/lib/libsherpa-ncnn-jni.so ]]; then
  mkdir -p ../build
  pushd ../build
  cmake \
    -DSHERPA_NCNN_ENABLE_PYTHON=OFF \
    -DSHERPA_NCNN_ENABLE_TESTS=OFF \
    -DSHERPA_NCNN_ENABLE_CHECK=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_NCNN_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_NCNN_ENABLE_JNI=ON \
    ..

  make -j4
  ls -lh lib
  popd
fi

export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

if [ ! -f ./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-ncnn/releases/download/asr-models/sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  tar xvf sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  rm sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
fi

out_filename=test_offline_sense_voice_asr.jar
kotlinc-jvm -include-runtime -d $out_filename \
  ./faked-asset-manager.kt \
  ./faked-log.kt \
  ./FeatureConfig.kt \
  ./OfflineRecognizer.kt \
  ./OfflineStream.kt \
  ./WaveReader.kt \
  \
  ./test_offline_sense_voice_asr.kt

ls -lh $out_filename

java -Djava.library.path=../build/lib -jar $out_filename
