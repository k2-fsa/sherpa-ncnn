#!/usr/bin/env bash

set -ex

if [ ! -d ./sherpa-ncnn-conv-emformer-transducer-2022-12-06 ]; then
  echo "Please download the pre-trained model for testing."
  echo "You can refer to"
  echo ""
  echo "https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/conv-emformer-transducer-models.html#csukuangfj-sherpa-ncnn-conv-emformer-transducer-2022-12-06-chinese-english"
  echo "for help"
  exit 1
fi

if [ ! -f ../build/lib/libsherpa-ncnn-core.a ]; then
  echo "Please build sherpa-ncnn first. You can use"
  echo ""
  echo "  cd /path/to/sherpa-ncnn"
  echo "  mkdir build"
  echo "  cd build"
  echo "  cmake .."
  echo "  make -j4"
  exit 1
fi

if [ ! -f ./sherpa-ncnn-ffmpeg ]; then
  make
fi

../ffmpeg-examples/sherpa-ncnn-ffmpeg \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/tokens.txt \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.param \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.bin \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.param \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.bin \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.param \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.bin \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/test_wavs/0.wav


../ffmpeg-examples/sherpa-ncnn-ffmpeg \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/tokens.txt \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.param \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.bin \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.param \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.bin \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.param \
  ./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.bin \
  https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-06/resolve/main/test_wavs/0.wav

