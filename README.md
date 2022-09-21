# Introduction

See <https://github.com/k2-fsa/sherpa>

This repo uses [ncnn](https://github.com/tencent/ncnn) for running the neural
network model and does not depend on libtorch.

Please read <https://k2-fsa.github.io/icefall/recipes/librispeech/lstm_pruned_stateless_transducer.html>
if you are interested in how the model is trained.

We provide exported models in ncnn format and they can be downloaded using
the following links:

- English: <https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-05>
- Chinese: (TODO, We will provide it soon)


# Usage


```bash
git clone https://github.com/k2-fsa/sherpa-ncnn
cd sherpa-ncnn
mkdir build
cd build
cmake ..
make -j
cd ..

# Now download the pretrained model for testing
git lfs install
git clone https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-05

./build/bin/sherpa-ncnn \
  ./sherpa-ncnn-2022-09-05/bar/tokens.txt \
  ./sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-05/test_wavs/1089-134686-0001.wav
```

[ncnn]: https://github.com/tencent/ncnn
