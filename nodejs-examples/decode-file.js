// Copyright (c)  2024  Xiaomi Corporation (authors: Fangjun Kuang)
const fs = require('fs');
const wav = require('wav');
const {Readable} = require('stream');

const sherpa_ncnn = require('sherpa-ncnn0');

function createRecognizer() {
  let modelConfig = {
    encoderParam:
        './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.param',
    encoderBin:
        './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.bin',
    decoderParam:
        './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.param',
    decoderBin:
        './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.bin',
    joinerParam:
        './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.param',
    joinerBin:
        './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.bin',
    tokens:
        './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/tokens.txt',
    useVulkanCompute: 0,
    numThreads: 1,
  };

  let decoderConfig = {
    decodingMethod: 'greedy_search',
    numActivePaths: 4,
  };

  let featConfig = {
    samplingRate: 16000,
    featureDim: 80,
  };

  let config = {
    featConfig: featConfig,
    modelConfig: modelConfig,
    decoderConfig: decoderConfig,
    enableEndpoint: 1,
    rule1MinTrailingSilence: 1.2,
    rule2MinTrailingSilence: 2.4,
    rule3MinUtternceLength: 20,
  };

  return sherpa_ncnn.createRecognizer(config);
}

const recognizer = createRecognizer();
const stream = recognizer.createStream();

console.log(recognizer.config);

const waveFilename =
    './sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/test_wavs/0.wav';

const reader = new wav.Reader();
const readable = new Readable().wrap(reader);
const buf = [];

reader.on('format', ({audioFormat, bitDepth, channels, sampleRate}) => {
  if (sampleRate != recognizer.config.featConfig.samplingRate) {
    throw new Error(`Only support sampleRate ${
        recognizer.config.featConfig.samplingRate}. Given ${sampleRate}`);
  }

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
          new Float32Array(recognizer.config.featConfig.samplingRate * 0.5);

      buf.push(floatSamples);
      const flattened =
          Float32Array.from(buf.reduce((a, b) => [...a, ...b], []));

      stream.acceptWaveform(
          recognizer.config.featConfig.samplingRate, flattened);
      while (recognizer.isReady(stream)) {
        recognizer.decode(stream);
      }
      const r = recognizer.getResult(stream);
      console.log('result:', r);

      stream.free();
      recognizer.free();
    });

readable.on('readable', function() {
  let chunk;
  while ((chunk = readable.read()) != null) {
    const int16Samples = new Int16Array(
        chunk.buffer, chunk.byteOffset,
        chunk.length / Int16Array.BYTES_PER_ELEMENT);

    const floatSamples = new Float32Array(int16Samples.length);
    for (let i = 0; i < floatSamples.length; i++) {
      floatSamples[i] = int16Samples[i] / 32768.0;
    }

    buf.push(floatSamples);
  }
});
