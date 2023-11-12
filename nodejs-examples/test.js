// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
//
const fs = require('fs');
const {Readable} = require('stream');
const wav = require('wav');

sherpa_ncnn = require('./index.js')

const featConfig = new sherpa_ncnn.FeatureConfig();
featConfig.sampleRate = 16000;
featConfig.featureDim = 80;

const decoderConfig = new sherpa_ncnn.DecoderConfig();
decoderConfig.decodingMethod = 'greedy_search';
decoderConfig.numActivePaths = 4;

const modelConfig = new sherpa_ncnn.ModelConfig();
modelConfig.encoderParam =
    './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.param';
modelConfig.encoderBin =
    './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.bin';

modelConfig.decoderParam =
    './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.param';
modelConfig.decoderBin =
    './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.bin';

modelConfig.joinerParam =
    './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.param';
modelConfig.joinerBin =
    './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.bin';

modelConfig.tokens =
    './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/tokens.txt';
modelConfig.useVulkanCompute = 0;
modelConfig.numThreads = 1;

const recognizerConfig = new sherpa_ncnn.RecognizerConfig();
recognizerConfig.featConfig = featConfig;
recognizerConfig.modelConfig = modelConfig;
recognizerConfig.decoderConfig = decoderConfig;

const recognizer = new sherpa_ncnn.Recognizer(recognizerConfig);

const waveFilename =
    './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/test_wavs/2.wav'

const reader = new wav.Reader();
const readable = new Readable().wrap(reader);

function decode(samples) {
  recognizer.acceptWaveform(recognizerConfig.featConfig.sampleRate, samples);

  while (recognizer.isReady()) {
    recognizer.decode();
  }
  const text = recognizer.getResult();
  console.log(text);
}

reader.on('format', ({audioFormat, sampleRate, channels, bitDepth}) => {
  if (audioFormat != 1) {
    throw new Error(`Only support PCM format. Given ${audioFormat}`);
  }
  if (channels != 1) {
    throw new Error(`Only a single channel. Given ${channel}`);
  }

  if (bitDepth != 16) {
    throw new Error(`Only support 16-bit samples. Given ${bitDepth}`);
  }
});

fs.createReadStream(waveFilename, {'highWaterMark': 4096})
    .pipe(reader)
    .on('finish', function(err) {
      // tail padding
      const floatSamples =
          new Float32Array(recognizerConfig.featConfig.sampleRate * 0.5);
      decode(floatSamples);
      recognizer.free()
    });


readable.on('readable', function() {
  let chunk;
  while ((chunk = readable.read()) != null) {
    const int16Samples = new Int16Array(
        chunk.buffer, chunk.byteOffset,
        chunk.length / Int16Array.BYTES_PER_ELEMENT);

    let floatSamples = new Float32Array(int16Samples.length);
    for (let i = 0; i < floatSamples.length; i++) {
      floatSamples[i] = int16Samples[i] / 32768.0;
    }
    decode(floatSamples);
  }
});
