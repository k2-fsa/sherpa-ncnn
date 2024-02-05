

function freeConfig(config) {
  if ('buffer' in config) {
    _free(config.buffer);
  }
  _free(config.ptr);
}

// The user should free the returned pointers
function initSherpaNcnnModelConfig(config) {
  let encoderParamLen = lengthBytesUTF8(config.encoderParam) + 1;
  let decoderParamLen = lengthBytesUTF8(config.decoderParam) + 1;
  let joinerParamLen = lengthBytesUTF8(config.joinerParam) + 1;

  let encoderBinLen = lengthBytesUTF8(config.encoderBin) + 1;
  let decoderBinLen = lengthBytesUTF8(config.decoderBin) + 1;
  let joinerBinLen = lengthBytesUTF8(config.joinerBin) + 1;

  let tokensLen = lengthBytesUTF8(config.tokens) + 1;

  let n = encoderParamLen + decoderParamLen + joinerParamLen;
  n += encoderBinLen + decoderBinLen + joinerBinLen;
  n += tokensLen;

  let buffer = _malloc(n);
  let ptr = _malloc(4 * 9);

  let offset = 0;
  stringToUTF8(config.encoderParam, buffer + offset, encoderParamLen);
  offset += encoderParamLen;

  stringToUTF8(config.encoderBin, buffer + offset, encoderBinLen);
  offset += encoderBinLen;

  stringToUTF8(config.decoderParam, buffer + offset, decoderParamLen);
  offset += decoderParamLen;

  stringToUTF8(config.decoderBin, buffer + offset, decoderBinLen);
  offset += decoderBinLen;

  stringToUTF8(config.joinerParam, buffer + offset, joinerParamLen);
  offset += joinerParamLen;

  stringToUTF8(config.joinerBin, buffer + offset, joinerBinLen);
  offset += joinerBinLen;

  stringToUTF8(config.tokens, buffer + offset, tokensLen);
  offset += tokensLen;

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');  // encoderParam
  offset += encoderParamLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');  // encoderBin
  offset += encoderBinLen;

  Module.setValue(ptr + 8, buffer + offset, 'i8*');  // decoderParam
  offset += decoderParamLen;

  Module.setValue(ptr + 12, buffer + offset, 'i8*');  // decoderBin
  offset += decoderBinLen;

  Module.setValue(ptr + 16, buffer + offset, 'i8*');  // joinerParam
  offset += joinerParamLen;

  Module.setValue(ptr + 20, buffer + offset, 'i8*');  // joinerBin
  offset += joinerBinLen;

  Module.setValue(ptr + 24, buffer + offset, 'i8*');  // tokens
  offset += tokensLen;

  Module.setValue(ptr + 28, config.useVulkanCompute, 'i32');
  Module.setValue(ptr + 32, config.numThreads, 'i32');

  return {
    buffer: buffer, ptr: ptr, len: 36,
  }
}

function initSherpaNcnnDecoderConfig(config) {
  let n = lengthBytesUTF8(config.decodingMethod) + 1;
  let buffer = _malloc(n);
  let ptr = _malloc(4 * 2);

  stringToUTF8(config.decodingMethod, buffer, n);

  Module.setValue(ptr, buffer, 'i8*');
  Module.setValue(ptr + 4, config.numActivePaths, 'i32');

  return {
    buffer: buffer, ptr: ptr, len: 8,
  }
}

function initSherpaNcnnFeatureExtractorConfig(config) {
  let ptr = _malloc(4 * 2);
  Module.setValue(ptr, config.samplingRate, 'float');
  Module.setValue(ptr + 4, config.featureDim, 'i32');
  return {
    ptr: ptr, len: 8,
  }
}

function initSherpaNcnnRecognizerConfig(config) {
  let featConfig = initSherpaNcnnFeatureExtractorConfig(config.featConfig);
  let modelConfig = initSherpaNcnnModelConfig(config.modelConfig);
  let decoderConfig = initSherpaNcnnDecoderConfig(config.decoderConfig);

  let numBytes =
      featConfig.len + modelConfig.len + decoderConfig.len + 4 * 4 + 4 * 2;

  let ptr = _malloc(numBytes);
  let offset = 0;
  _CopyHeap(featConfig.ptr, featConfig.len, ptr + offset);
  offset += featConfig.len;

  _CopyHeap(modelConfig.ptr, modelConfig.len, ptr + offset)
  offset += modelConfig.len;

  _CopyHeap(decoderConfig.ptr, decoderConfig.len, ptr + offset)
  offset += decoderConfig.len;

  Module.setValue(ptr + offset, config.enableEndpoint, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.rule1MinTrailingSilence, 'float');
  offset += 4;

  Module.setValue(ptr + offset, config.rule2MinTrailingSilence, 'float');
  offset += 4;

  Module.setValue(ptr + offset, config.rule3MinUtternceLength, 'float');
  offset += 4;

  Module.setValue(ptr + offset, 0, 'i32');  // hotwords file
  offset += 4;

  Module.setValue(ptr + offset, 0.5, 'float');  // hotwords_score
  offset += 4;

  return {
    ptr: ptr, len: numBytes, featConfig: featConfig, modelConfig: modelConfig,
        decoderConfig: decoderConfig,
  }
}

class Stream {
  constructor(handle) {
    this.handle = handle;
    this.pointer = null;
    this.n = 0
  }

  free() {
    if (this.handle) {
      _DestroyStream(this.handle);
      this.handle = null;
      _free(this.pointer);
      this.pointer = null;
      this.n = 0;
    }
  }

  /**
   * @param sampleRate {Number}
   * @param samples {Float32Array} Containing samples in the range [-1, 1]
   */
  acceptWaveform(sampleRate, samples) {
    if (this.n < samples.length) {
      _free(this.pointer);
      this.pointer = _malloc(samples.length * samples.BYTES_PER_ELEMENT);
      this.n = samples.length
    }

    Module.HEAPF32.set(samples, this.pointer / samples.BYTES_PER_ELEMENT);
    _AcceptWaveform(this.handle, sampleRate, this.pointer, samples.length);
  }

  inputFinished() {
    _InputFinished(this.handle);
  }
};

class Recognizer {
  constructor(configObj, borrowedHandle) {
    if (borrowedHandle) {
      this.handle = borrowedHandle;
      return;
    }

    let config = initSherpaNcnnRecognizerConfig(configObj)
    let handle = _CreateRecognizer(config.ptr);

    freeConfig(config.featConfig);
    freeConfig(config.modelConfig);
    freeConfig(config.decoderConfig);
    freeConfig(config);

    this.handle = handle;
  }

  free() {
    _DestroyRecognizer(this.handle);
    this.handle = 0
  }

  createStream() {
    let handle = _CreateStream(this.handle);
    return new Stream(handle);
  }

  isReady(stream) {
    return _IsReady(this.handle, stream.handle) == 1;
  }

  isEndpoint(stream) {
    return _IsEndpoint(this.handle, stream.handle) == 1;
  }

  decode(stream) {
    return _Decode(this.handle, stream.handle);
  }

  reset(stream) {
    _Reset(this.handle, stream.handle);
  }

  getResult(stream) {
    let r = _GetResult(this.handle, stream.handle);
    let textPtr = getValue(r, 'i8*');
    let text = UTF8ToString(textPtr);
    _DestroyResult(r);
    return text;
  }
}

function createRecognizer() {
  let modelConfig = {
    encoderParam: './encoder_jit_trace-pnnx.ncnn.param',
    encoderBin: './encoder_jit_trace-pnnx.ncnn.bin',
    decoderParam: './decoder_jit_trace-pnnx.ncnn.param',
    decoderBin: './decoder_jit_trace-pnnx.ncnn.bin',
    joinerParam: './joiner_jit_trace-pnnx.ncnn.param',
    joinerBin: './joiner_jit_trace-pnnx.ncnn.bin',
    tokens: './tokens.txt',
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

  let configObj = {
    featConfig: featConfig,
    modelConfig: modelConfig,
    decoderConfig: decoderConfig,
    enableEndpoint: 1,
    rule1MinTrailingSilence: 1.2,
    rule2MinTrailingSilence: 2.4,
    rule3MinUtternceLength: 20,
  };

  return new Recognizer(configObj);
}
