# Introduction

This folder contains examples about how to use the sherpa-ncnn WebAssembly module
with nodejs for speech recognition.

- [decode-file.js](./decode-file.js) it shows how to decode a file

## Usage

### Install dependencies

```bash
cd ./nodejs-wasm-examples
npm i
```

### Download a model

Please visit <https://github.com/k2-fsa/sherpa-ncnn/releases/tag/models> to
select more models.

The following is an example:

```bash
cd ./nodejs-wasm-examples
wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/models/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13.tar.bz2
tar xvf sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13.tar.bz2
rm sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13.tar.bz2
```

### Decode a file

```bash
node ./decode-file.js
```

### Real-time speech recognition from a microphone

```bash
node ./real-time-speech-recognition-microphone.js
```
