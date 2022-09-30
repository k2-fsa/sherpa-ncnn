# Introduction

**Documentation**: <https://k2-fsa.github.io/sherpa/ncnn/index.html>

See <https://github.com/k2-fsa/sherpa>

This repo uses [ncnn](https://github.com/tencent/ncnn) for running the neural
network model and does not depend on libtorch.

Please read <https://k2-fsa.github.io/icefall/recipes/librispeech/lstm_pruned_stateless_transducer.html>
if you are interested in how the model is trained.

We provide exported models in ncnn format and they can be downloaded using
the following links:

- English: <https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-05>
- Chinese: <https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-30>


# Usage


```bash
git clone https://github.com/k2-fsa/sherpa-ncnn
cd sherpa-ncnn
mkdir build
cd build
cmake ..
make -j6
cd ..
```

## Download the pretrained model (Chinese)

**Caution**: You have to run `git lfs install`. Otherwise, you will be **SAD** later.

```bash
git lfs install
git clone https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-30

./build/bin/sherpa-ncnn \
  ./sherpa-ncnn-2022-09-30/tokens.txt \
  ./sherpa-ncnn-2022-09-30/encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-30/encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-30/decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-30/decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-30/joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-30/joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-30/test_wavs/0.wav
```

To do speech recognition in real-time with a microphone, run:

```bash
./build/bin/sherpa-ncnn-microphone \
  ./sherpa-ncnn-2022-09-30/tokens.txt \
  ./sherpa-ncnn-2022-09-30/encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-30/encoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-30/decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-30/decoder_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-30/joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-30/joiner_jit_trace-epoch-11-avg-2-pnnx.ncnn.bin
```

## Download the pretrained model (English)

```bash
git lfs install
git clone https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-05

./build/bin/sherpa-ncnn \
  ./sherpa-ncnn-2022-09-05/tokens.txt \
  ./sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-05/test_wavs/1089-134686-0001.wav
```

To do speech recognition in real-time with a microphone, run:

```bash
./build/bin/sherpa-ncnn-microphone \
  ./sherpa-ncnn-2022-09-05/tokens.txt \
  ./sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
  ./sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
  ./sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin
```


[ncnn]: https://github.com/tencent/ncnn
