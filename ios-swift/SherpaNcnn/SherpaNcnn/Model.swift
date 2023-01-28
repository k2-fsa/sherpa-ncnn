
import Foundation
/// https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/conv-emformer-transducer-models.html#csukuangfj-sherpa-ncnn-conv-emformer-transducer-2022-12-06-chinese-english
func getMultiLingualModelConfig2022_12_06() -> SherpaNcnnModelConfig{
    let encoderParam = Bundle.main.path(forResource: "encoder_jit_trace-pnnx.ncnn", ofType: "param")
    let encoderBin = Bundle.main.path(forResource: "encoder_jit_trace-pnnx.ncnn", ofType: "bin")
    let decoderParam = Bundle.main.path(forResource: "decoder_jit_trace-pnnx.ncnn", ofType: "param")
    let decoderBin = Bundle.main.path(forResource: "decoder_jit_trace-pnnx.ncnn", ofType: "bin")
    let joinerParam = Bundle.main.path(forResource: "joiner_jit_trace-pnnx.ncnn", ofType: "param")
    let joinerBin = Bundle.main.path(forResource: "joiner_jit_trace-pnnx.ncnn", ofType: "bin")
    let tokens = Bundle.main.path(forResource: "tokens", ofType: "txt")

    return sherpaNcnnModelConfig(encoderParam: encoderParam!,
                                 encoderBin: encoderBin!,
                                 decoderParam: decoderParam!,
                                 decoderBin: decoderBin!,
                                 joinerParam: joinerParam!,
                                 joinerBin: joinerBin!,
                                 tokens: tokens!,
                                 numThreads: 4
    )
}
