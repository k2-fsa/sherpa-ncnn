#!/usr/bin/env bash

dotnet build -c Release --verbosity normal

# Please refer to
# https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/zipformer-transucer-models.html#csukuangfj-sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13-bilingual-chinese-english
# to download the model files

if [ ! -d ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13 ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13
  cd sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13
  git lfs pull --include "*.bin"

  cd ..
fi

dotnet run -c Release \
  ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/tokens.txt \
  ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.param \
  ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.bin \
  ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.param \
  ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.bin \
  ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.param \
  ./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.bin \
  1 \
  greedy_search
