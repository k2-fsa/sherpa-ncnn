# Introduction

Please refer to
https://github.com/k2-fsa/sherpa-ncnn/releases/tag/models
to download a model.

The following is an example:
```
cd /path/to/this/directory
wget -q https://github.com/k2-fsa/sherpa-ncnn/releases/download/models/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13.tar.bz2
tar xf sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13.tar.bz2
rm sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13.tar.bz2
mv sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/*pnnx.ncnn.param .
mv sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/*pnnx.ncnn.bin .
mv sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/tokens.txt .
```

You should have the following files in `assets` before you can run
`build-wasm-simd.sh`

```
assets fangjun$ tree .
.
├── README.md
├── decoder_jit_trace-pnnx.ncnn.bin
├── decoder_jit_trace-pnnx.ncnn.param
├── encoder_jit_trace-pnnx.ncnn.bin
├── encoder_jit_trace-pnnx.ncnn.param
├── joiner_jit_trace-pnnx.ncnn.bin
├── joiner_jit_trace-pnnx.ncnn.param
└── tokens.txt

0 directories, 8 files
```
