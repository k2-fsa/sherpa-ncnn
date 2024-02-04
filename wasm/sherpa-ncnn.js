

function freeConfig(config, wasmModule) {
  if ('buffer' in config) {
    wasmModule._free(config.buffer);
  }
  wasmModule._free(config.ptr);
}

// The user should free the returned pointers
function initSherpaNcnnModelConfig(config, wasmModule) {
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

  let buffer = wasmModule._malloc(n);
  let ptr = wasmModule._malloc(4 * 9);

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

function initSherpaNcnnDecoderConfig(config, wasmModule) {
  let n = lengthBytesUTF8(config.decodingMethod) + 1;
  let buffer = wasmModule._malloc(n);
  let ptr = wasmModule._malloc(4 * 2);

  stringToUTF8(config.decodingMethod, buffer, n);

  Module.setValue(ptr, buffer, 'i8*');
  Module.setValue(ptr + 4, config.numActivePaths, 'i32');

  return {
    buffer: buffer, ptr: ptr, len: 8,
  }
}

function initSherpaNcnnFeatureExtractorConfig(config, wasmModule) {
  let ptr = wasmModule._malloc(4 * 2);
  Module.setValue(ptr, config.samplingRate, 'float');
  Module.setValue(ptr + 4, config.featureDim, 'i32');
  return {
    ptr: ptr, len: 8,
  }
}

function copyHeap(dstPtr, srcPtr, len, wasmModule) {
  console.log(len);

  let src = new Uint8Array(HEAPU8, srcPtr, len);
  let dst = new Uint8Array(HEAPU8, dstPtr, len);
  dst.set(src)
}

function initSherpaNcnnRecognizerConfig(config, wasmModule) {
  let featConfig =
      initSherpaNcnnFeatureExtractorConfig(config.featConfig, wasmModule);
  let modelConfig = initSherpaNcnnModelConfig(config.modelConfig, wasmModule);
  let decoderConfig =
      initSherpaNcnnDecoderConfig(config.decoderConfig, wasmModule);

  let numBytes =
      featConfig.len + modelConfig.len + decoderConfig.len + 4 * 4 + 4 * 2;
  console.log(numBytes)

  let ptr = wasmModule._malloc(numBytes);
  let offset = 0;
  wasmModule._CopyHeap(featConfig.ptr, featConfig.len, ptr + offset);
  offset += featConfig.len;

  wasmModule._CopyHeap(modelConfig.ptr, modelConfig.len, ptr + offset)
  offset += modelConfig.len;

  wasmModule._CopyHeap(decoderConfig.ptr, decoderConfig.len, ptr + offset)
  offset += decoderConfig.len;

  wasmModule.setValue(ptr + offset, config.enableEndpoint, 'i32');
  offset += 4;

  wasmModule.setValue(ptr + offset, config.rule1MinTrailingSilence, 'float');
  offset += 4;

  wasmModule.setValue(ptr + offset, config.rule2MinTrailingSilence, 'float');
  offset += 4;

  wasmModule.setValue(ptr + offset, config.rule3MinUtternceLength, 'float');
  offset += 4;

  wasmModule.setValue(ptr + offset, 0, 'i32');  // hotwords file
  offset += 4;

  wasmModule.setValue(ptr + offset, 0.5, 'float');  // hotwords_score
  offset += 4;

  freeConfig(featConfig, wasmModule);
  freeConfig(modelConfig, wasmModule);
  freeConfig(decoderConfig, wasmModule);

  return {
    ptr: ptr, len: numBytes,
  }
}

class Recognizer {
  constructor(config, wasmModule) {
    modelConfig = initSherpaNcnnModelConfig(config.modelConfig)
  }
}
