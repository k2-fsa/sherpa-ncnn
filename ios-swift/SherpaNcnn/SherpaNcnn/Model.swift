import Foundation

func getResource(_ forResource: String, _ ofType: String) -> String {
  let path = Bundle.main.path(forResource: forResource, ofType: ofType)
  precondition(
    path != nil,
    "\(forResource).\(ofType) does not exist!\n" + "Remember to change \n"
      + "  Build Phases -> Copy Bundle Resources\n" + "to add it!"
  )
  return path!
}
/// Please refer to
/// https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
/// to download pre-trained models

/// csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-06 (Chinese + English)
func getMultilingualModelConfig2022_12_06() -> SherpaNcnnModelConfig {
  let encoderParam = getResource("encoder_jit_trace-pnnx.ncnn", "param")
  let encoderBin = getResource("encoder_jit_trace-pnnx.ncnn", "bin")
  let decoderParam = getResource("decoder_jit_trace-pnnx.ncnn", "param")
  let decoderBin = getResource("decoder_jit_trace-pnnx.ncnn", "bin")
  let joinerParam = getResource("joiner_jit_trace-pnnx.ncnn", "param")
  let joinerBin = getResource("joiner_jit_trace-pnnx.ncnn", "bin")
  let tokens = getResource("tokens", "txt")

  return sherpaNcnnModelConfig(
    encoderParam: encoderParam,
    encoderBin: encoderBin,
    decoderParam: decoderParam,
    decoderBin: decoderBin,
    joinerParam: joinerParam,
    joinerBin: joinerBin,
    tokens: tokens,
    numThreads: 4
  )
}

/// csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-06 (Chinese + English)
func getMultilingualModelConfig2022_12_06_Int8() -> SherpaNcnnModelConfig {
  let encoderParam = getResource("encoder_jit_trace-pnnx.ncnn.int8", "param")
  let encoderBin = getResource("encoder_jit_trace-pnnx.ncnn.int8", "bin")
  let decoderParam = getResource("decoder_jit_trace-pnnx.ncnn", "param")
  let decoderBin = getResource("decoder_jit_trace-pnnx.ncnn", "bin")
  let joinerParam = getResource("joiner_jit_trace-pnnx.ncnn.int8", "param")
  let joinerBin = getResource("joiner_jit_trace-pnnx.ncnn.int8", "bin")
  let tokens = getResource("tokens", "txt")

  return sherpaNcnnModelConfig(
    encoderParam: encoderParam,
    encoderBin: encoderBin,
    decoderParam: decoderParam,
    decoderBin: decoderBin,
    joinerParam: joinerParam,
    joinerBin: joinerBin,
    tokens: tokens,
    numThreads: 4
  )
}

/// marcoyang/sherpa-ncnn-conv-emformer-transducer-small-2023-01-09 (English)
func getConvEmformerSmallEnglishModelConfig2023_01_09() -> SherpaNcnnModelConfig {
  let encoderParam = getResource("encoder_jit_trace-pnnx.ncnn", "param")
  let encoderBin = getResource("encoder_jit_trace-pnnx.ncnn", "bin")
  let decoderParam = getResource("decoder_jit_trace-pnnx.ncnn", "param")
  let decoderBin = getResource("decoder_jit_trace-pnnx.ncnn", "bin")
  let joinerParam = getResource("joiner_jit_trace-pnnx.ncnn", "param")
  let joinerBin = getResource("joiner_jit_trace-pnnx.ncnn", "bin")
  let tokens = getResource("tokens", "txt")

  return sherpaNcnnModelConfig(
    encoderParam: encoderParam,
    encoderBin: encoderBin,
    decoderParam: decoderParam,
    decoderBin: decoderBin,
    joinerParam: joinerParam,
    joinerBin: joinerBin,
    tokens: tokens,
    numThreads: 4
  )
}

/// marcoyang/sherpa-ncnn-conv-emformer-transducer-small-2023-01-09 (English)
func getConvEmformerSmallEnglishModelConfig2023_01_09_Int8() -> SherpaNcnnModelConfig {
  let encoderParam = getResource("encoder_jit_trace-pnnx.ncnn.int8", "param")
  let encoderBin = getResource("encoder_jit_trace-pnnx.ncnn.int8", "bin")
  let decoderParam = getResource("decoder_jit_trace-pnnx.ncnn", "param")
  let decoderBin = getResource("decoder_jit_trace-pnnx.ncnn", "bin")
  let joinerParam = getResource("joiner_jit_trace-pnnx.ncnn.int8", "param")
  let joinerBin = getResource("joiner_jit_trace-pnnx.ncnn.int8", "bin")
  let tokens = getResource("tokens", "txt")

  return sherpaNcnnModelConfig(
    encoderParam: encoderParam,
    encoderBin: encoderBin,
    decoderParam: decoderParam,
    decoderBin: decoderBin,
    joinerParam: joinerParam,
    joinerBin: joinerBin,
    tokens: tokens,
    numThreads: 4
  )
}
/// marcoyang/sherpa-ncnn-conv-emformer-transducer-small-2023-01-09 (English)
func getLstmTransducerEnglish_2022_09_05() -> SherpaNcnnModelConfig {
  let encoderParam = getResource(
    "encoder_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn", "param")
  let encoderBin = getResource(
    "encoder_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn", "bin")
  let decoderParam = getResource(
    "decoder_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn", "param")
  let decoderBin = getResource(
    "decoder_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn", "bin")
  let joinerParam = getResource(
    "joiner_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn", "param")
  let joinerBin = getResource(
    "joiner_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn", "bin")
  let tokens = getResource("tokens", "txt")

  return sherpaNcnnModelConfig(
    encoderParam: encoderParam,
    encoderBin: encoderBin,
    decoderParam: decoderParam,
    decoderBin: decoderBin,
    joinerParam: joinerParam,
    joinerBin: joinerBin,
    tokens: tokens,
    numThreads: 4
  )
}

/// Please refer to
/// https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
/// to add more models if you need
