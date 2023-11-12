// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
//
// Please use
//
// npm install ffi-napi ref-struct-napi
//
// before you use this file
//
//
// Please use node 16. node 18, 20, and 21 are known not working.
// See also
// https://github.com/node-ffi-napi/node-ffi-napi/issues/244
'use strict'

const debug = require('debug')('sherpa-ncnn');
const os = require('os');
const path = require('path');
const ffi = require('ffi-napi');
const ref = require('ref-napi');
const fs = require('fs');

const StructType = require('ref-struct-napi');
const cstring = ref.types.CString;
const int32_t = ref.types.int32;
const float = ref.types.float;
const floatPtr = ref.refType(float);

const RecognizerPtr = ref.refType(ref.types.void);
const StreamPtr = ref.refType(ref.types.void);
const SherpaNcnnModelConfig = StructType({
  'encoderParam': cstring,
  'encoderBin': cstring,
  'decoderParam': cstring,
  'decoderBin': cstring,
  'joinerParam': cstring,
  'joinerBin': cstring,
  'tokens': cstring,
  'useVulkanCompute': int32_t,
  'numThreads': int32_t,
});

const SherpaNcnnDecoderConfig = StructType({
  'decodingMethod': cstring,
  'numActivePaths': int32_t,
});

const SherpaNcnnFeatureExtractorConfig = StructType({
  'sampleRate': float,
  'featureDim': int32_t,
});

const SherpaNcnnRecognizerConfig = StructType({
  'featConfig': SherpaNcnnFeatureExtractorConfig,
  'modelConfig': SherpaNcnnModelConfig,
  'decoderConfig': SherpaNcnnDecoderConfig,
  'enableEndpoint': int32_t,
  'rule1MinTrailingSilence': float,
  'rule2MinTrailingSilence': float,
  'rule3MinUtteranceLength': float,
  'hotwordsFile': cstring,
  'hotwordsScore': cstring,
});

const SherpaNcnnResult = StructType({
  'text': cstring,
  'tokens': cstring,
  'timestamps': floatPtr,
  'count': int32_t,
});


const ResultPtr = ref.refType(SherpaNcnnResult);
const RecognizerConfigPtr = ref.refType(SherpaNcnnRecognizerConfig)

let soname;
if (os.platform() == 'win32') {
  soname = path.join(__dirname, 'install', 'lib', 'sherpa-ncnn-c-api.dll');
} else if (os.platform() == 'darwin') {
  soname = path.join(__dirname, 'install', 'lib', 'libsherpa-ncnn-c-api.dylib');
} else if (os.platform() == 'linux') {
  soname = path.join(__dirname, 'install', 'lib', 'libsherpa-ncnn-c-api.so');
} else {
  throw new Error(`Unsupported platform ${os.platform()}`);
}
if (!fs.existsSync(soname)) {
  throw new Error(`Cannot find file ${soname}. Please make sure you have run
      ./build.sh`);
}

debug('soname ', soname)

const libsherpa_ncnn = ffi.Library(soname, {
  'CreateRecognizer': [RecognizerPtr, [RecognizerConfigPtr]],
  'DestroyRecognizer': ['void', [RecognizerPtr]],
  'CreateStream': [StreamPtr, [RecognizerPtr]],
  'DestroyStream': ['void', [StreamPtr]],
  'AcceptWaveform': ['void', [StreamPtr, float, floatPtr, int32_t]],
  'IsReady': [int32_t, [RecognizerPtr, StreamPtr]],
  'Decode': ['void', [RecognizerPtr, StreamPtr]],
  'GetResult': [ResultPtr, [RecognizerPtr, StreamPtr]],
  'DestroyResult': ['void', [ResultPtr]],
  'Reset': ['void', [RecognizerPtr, StreamPtr]],
  'InputFinished': ['void', [StreamPtr]],
  'IsEndpoint': [int32_t, [RecognizerPtr, StreamPtr]],
});

class Recognizer {
  /**
   * @param {SherpaNcnnRecognizerConfig} config Configuration for the recognizer
   *
   * The user has to invoke this.free() at the end to avoid memory leak.
   */
  constructor(config) {
    this.recognizer_handle = libsherpa_ncnn.CreateRecognizer(config.ref());
    this.stream_handle = libsherpa_ncnn.CreateStream(this.recognizer_handle);
  }

  free() {
    if (this.stream_handle) {
      libsherpa_ncnn.DestroyStream(this.stream_handle);
      this.stream_handle = null;
    }

    libsherpa_ncnn.DestroyRecognizer(this.recognizer_handle);
    this.handle = null;
  }

  /**
   * @param {bool} true to create a new stream
   */
  reset(recreate) {
    if (recreate) {
      libsherpa_ncnn.DestroyStream(this.stream_handle);
      this.stream_handle = libsherpa_ncnn.CreateStream(this.recognizer_handle);
      return;
    }
    libsherpa_ncnn.Reset(this.recognizer_handle, this.stream_handle)
  }
  /**
   * @param {float} Sample rate of the input data
   * @param {float[]} A 1-d float array containing audio samples. It should be
   *                  in the range [-1, 1].
   */
  acceptWaveform(sampleRate, samples) {
    libsherpa_ncnn.AcceptWaveform(
        this.stream_handle, sampleRate, samples, samples.length);
  }

  isReady() {
    return libsherpa_ncnn.IsReady(this.recognizer_handle, this.stream_handle);
  }

  decode() {
    libsherpa_ncnn.Decode(this.recognizer_handle, this.stream_handle);
  }

  getResult() {
    const h =
        libsherpa_ncnn.GetResult(this.recognizer_handle, this.stream_handle);
    const text = Buffer.from(h.deref().text, 'utf-8').toString();

    // TODO(fangjun): Enable it to avoid memory leak.
    //
    // It is commented out since it causes errors occasionally.
    //
    // libsherpa_ncnn.DestroyResult(h);
    return text;
  }
};

// alias

const ModelConfig = SherpaNcnnModelConfig;
const DecoderConfig = SherpaNcnnDecoderConfig;
const FeatureConfig = SherpaNcnnFeatureExtractorConfig;
const RecognizerConfig = SherpaNcnnRecognizerConfig;

module.exports = {
  FeatureConfig,
  ModelConfig,
  DecoderConfig,
  Recognizer,
  RecognizerConfig,
};
