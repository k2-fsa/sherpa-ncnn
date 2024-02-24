

function freeConfig(config, Module) {
  if ('buffer' in config) {
    Module._free(config.buffer);
  }
  Module._free(config.ptr);
}

// The user should free the returned pointers
function initSherpaNcnnModelConfig(config, Module) {
  let encoderParamLen = Module.lengthBytesUTF8(config.encoderParam) + 1;
  let decoderParamLen = Module.lengthBytesUTF8(config.decoderParam) + 1;
  let joinerParamLen = Module.lengthBytesUTF8(config.joinerParam) + 1;

  let encoderBinLen = Module.lengthBytesUTF8(config.encoderBin) + 1;
  let decoderBinLen = Module.lengthBytesUTF8(config.decoderBin) + 1;
  let joinerBinLen = Module.lengthBytesUTF8(config.joinerBin) + 1;

  let tokensLen = Module.lengthBytesUTF8(config.tokens) + 1;

  let n = encoderParamLen + decoderParamLen + joinerParamLen;
  n += encoderBinLen + decoderBinLen + joinerBinLen;
  n += tokensLen;

  let buffer = Module._malloc(n);
  let ptr = Module._malloc(4 * 9);

  let offset = 0;
  Module.stringToUTF8(config.encoderParam, buffer + offset, encoderParamLen);
  offset += encoderParamLen;

  Module.stringToUTF8(config.encoderBin, buffer + offset, encoderBinLen);
  offset += encoderBinLen;

  Module.stringToUTF8(config.decoderParam, buffer + offset, decoderParamLen);
  offset += decoderParamLen;

  Module.stringToUTF8(config.decoderBin, buffer + offset, decoderBinLen);
  offset += decoderBinLen;

  Module.stringToUTF8(config.joinerParam, buffer + offset, joinerParamLen);
  offset += joinerParamLen;

  Module.stringToUTF8(config.joinerBin, buffer + offset, joinerBinLen);
  offset += joinerBinLen;

  Module.stringToUTF8(config.tokens, buffer + offset, tokensLen);
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

function initSherpaNcnnDecoderConfig(config, Module) {
  let n = Module.lengthBytesUTF8(config.decodingMethod) + 1;
  let buffer = Module._malloc(n);
  let ptr = Module._malloc(4 * 2);

  Module.stringToUTF8(config.decodingMethod, buffer, n);

  Module.setValue(ptr, buffer, 'i8*');
  Module.setValue(ptr + 4, config.numActivePaths, 'i32');

  return {
    buffer: buffer, ptr: ptr, len: 8,
  }
}

function initSherpaNcnnFeatureExtractorConfig(config, Module) {
  let ptr = Module._malloc(4 * 2);
  Module.setValue(ptr, config.samplingRate, 'float');
  Module.setValue(ptr + 4, config.featureDim, 'i32');
  return {
    ptr: ptr, len: 8,
  }
}

function initSherpaNcnnRecognizerConfig(config, Module) {
  let featConfig =
      initSherpaNcnnFeatureExtractorConfig(config.featConfig, Module);
  let modelConfig = initSherpaNcnnModelConfig(config.modelConfig, Module);
  let decoderConfig = initSherpaNcnnDecoderConfig(config.decoderConfig, Module);

  let numBytes =
      featConfig.len + modelConfig.len + decoderConfig.len + 4 * 4 + 4 * 2;

  let ptr = Module._malloc(numBytes);
  let offset = 0;
  Module._CopyHeap(featConfig.ptr, featConfig.len, ptr + offset);
  offset += featConfig.len;

  Module._CopyHeap(modelConfig.ptr, modelConfig.len, ptr + offset)
  offset += modelConfig.len;

  Module._CopyHeap(decoderConfig.ptr, decoderConfig.len, ptr + offset)
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
  constructor(handle, Module) {
    this.handle = handle;
    this.pointer = null;
    this.n = 0;
    this.Module = Module;
  }

  free() {
    if (this.handle) {
      this.Module._DestroyStream(this.handle);
      this.handle = null;
      this.Module._free(this.pointer);
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
      this.Module._free(this.pointer);
      this.pointer =
          this.Module._malloc(samples.length * samples.BYTES_PER_ELEMENT);
      this.n = samples.length
    }

    this.Module.HEAPF32.set(samples, this.pointer / samples.BYTES_PER_ELEMENT);
    this.Module._AcceptWaveform(
        this.handle, sampleRate, this.pointer, samples.length);
  }

  inputFinished() {
    _InputFinished(this.handle);
  }
};

class Recognizer {
  constructor(configObj, Module) {
    this.config = configObj;
    let config = initSherpaNcnnRecognizerConfig(configObj, Module)
    let handle = Module._CreateRecognizer(config.ptr);

    freeConfig(config.featConfig, Module);
    freeConfig(config.modelConfig, Module);
    freeConfig(config.decoderConfig, Module);
    freeConfig(config, Module);

    this.handle = handle;
    this.Module = Module;
  }

  free() {
    this.Module._DestroyRecognizer(this.handle);
    this.handle = 0
  }

  createStream() {
    let handle = this.Module._CreateStream(this.handle);
    return new Stream(handle, this.Module);
  }

  isReady(stream) {
    return this.Module._IsReady(this.handle, stream.handle) == 1;
  }

  isEndpoint(stream) {
    return this.Module._IsEndpoint(this.handle, stream.handle) == 1;
  }

  decode(stream) {
    return this.Module._Decode(this.handle, stream.handle);
  }

  reset(stream) {
    this.Module._Reset(this.handle, stream.handle);
  }

  getResult(stream) {
    let r = this.Module._GetResult(this.handle, stream.handle);
    let textPtr = this.Module.getValue(r, 'i8*');
    let text = this.Module.UTF8ToString(textPtr);
    this.Module._DestroyResult(r);
    return text;
  }
}

function createRecognizer(Module, myConfig) {
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

  if (myConfig) {
    configObj = myConfig;
  }

  return new Recognizer(configObj, Module);
}

if (typeof process == 'object' && typeof process.versions == 'object' &&
    typeof process.versions.node == 'string') {
  module.exports = {
    createRecognizer,
  };
}
