#!/usr/bin/env bash

set -e

# Please download the torchscript model from
# https://huggingface.co/pfluo/k2fsa-zipformer-chinese-english-mixed

if [ ! -d bilingual ]; then
  mkdir -p bilingual
  pushd bilingual
  ln -s ~/open-source/icefall-models/k2fsa-zipformer-chinese-english-mixed/exp/pretrained.pt epoch-99.pt
  ln -s ~/open-source/icefall-models/k2fsa-zipformer-chinese-english-mixed/data .
  popd
fi

./pruned_transducer_stateless7_streaming/export-for-ncnn-zh.py \
  --lang-dir ./bilingual/data/lang_char_bpe \
  --exp-dir ./bilingual \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --decode-chunk-len 32 \
  --num-encoder-layers "2,4,3,2,4" \
  --feedforward-dims "1024,1024,1536,1536,1024" \
  --nhead "8,8,8,8,8" \
  --encoder-dims "384,384,384,384,384" \
  --attention-dims "192,192,192,192,192" \
  --encoder-unmasked-dims "256,256,256,256,256" \
  --zipformer-downsampling-factors "1,2,4,8,2" \
  --cnn-module-kernels "31,31,31,31,31" \
  --decoder-dim 512 \
  --joiner-dim 512

cd bilingual

pnnx encoder_jit_trace-pnnx.pt
pnnx decoder_jit_trace-pnnx.pt
pnnx joiner_jit_trace-pnnx.pt
# TODO: modifiy encoder_jit_trace.ncnn.param to support sherpa-ncnn
