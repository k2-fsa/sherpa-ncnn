

function freeConfig(config, wasmModule) {
  wasmModule._free(config.buffer);
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
    buffer: buffer, ptr: ptr,
  }
}

function initSherpaNcnnDecoderConfig(config, wasmModule) {
  let n = lengthBytesUTF8(config.decodingMethod) + 1;
  let buffer = wasmModule._malloc(n);
  let ptr = wasmModule._malloc(4 * 2);
}

class Recognizer {
  constructor(config, wasmModule) {
    modelConfig = initSherpaNcnnModelConfig(config.modelConfig)
  }
}
