//
//  SherpaNcnn.swift
//  SherpaNcnn
//
//  Created by fangjun on 2023/1/28.
//

import Foundation

func toCPointer(_ s: String) -> UnsafePointer<Int8>! {
    let cs = (s as NSString).utf8String
    return UnsafePointer<Int8>(cs)
}

func sherpaNcnnModelConfig(encoderParam: String,
                           encoderBin: String,
                           decoderParam: String,
                           decoderBin: String,
                           joinerParam: String,
                           joinerBin: String,
                           tokens: String,
                           numThreads: Int = 4,
                           useVulkanCompute: Bool = true) -> SherpaNcnnModelConfig{
    return SherpaNcnnModelConfig(encoder_param: toCPointer(encoderParam),
                                 encoder_bin: toCPointer(encoderBin),
                                 decoder_param: toCPointer(decoderParam),
                                 decoder_bin: toCPointer(decoderBin),
                                 joiner_param: toCPointer(joinerParam),
                                 joiner_bin: toCPointer(joinerBin),
                                 tokens: toCPointer(tokens),
                                 use_vulkan_compute: useVulkanCompute ? 1 : 0,
                                 num_threads: Int32(numThreads))
    
}

func sherpaNcnnDecoderConfig(decodingMethod: String = "greedy_search",
                             numActivePaths: Int = 4,
                             enableEndpoint: Bool = false,
                             rule1MinTrailingSilence: Float = 2.4,
                             rule2MinTrailingSilence: Float = 1.2,
                             rule3MinUtteranceLength: Float = 30
)->SherpaNcnnDecoderConfig{
    return SherpaNcnnDecoderConfig(decoding_method: toCPointer(decodingMethod),
                                   num_active_paths: Int32(numActivePaths),
                                   enable_endpoint: enableEndpoint ? 1 : 0,
                                   rule1_min_trailing_silence: rule1MinTrailingSilence,
                                   rule2_min_trailing_silence: rule2MinTrailingSilence,
                                   rule3_min_utterance_length: rule3MinUtteranceLength)
}

class SherpaNcnnRecongitionResult {
    let result: UnsafePointer<SherpaNcnnResult>!
    
    var text: String {
        return String(cString: result.pointee.text)
    }
    
    init(result: UnsafePointer<SherpaNcnnResult>!) {
        self.result = result
    }
    
    deinit {
        if let result {
            DestroyResult(result)
        }
    }
}

class SherpaNcnnRecognizer {
    let recognizer: OpaquePointer!
    
    init(modelConfig: UnsafePointer<SherpaNcnnModelConfig>!,
         decoderConfig: UnsafePointer<SherpaNcnnDecoderConfig>!) {
        recognizer = CreateRecognizer(modelConfig, decoderConfig)
    }
    
    deinit {
        if let recognizer {
            DestroyRecognizer(recognizer)
        }
    }
    
    func acceptWaveform(samples: [Float], sampleRate: Float = 16000) -> Void {
        AcceptWaveform(recognizer, sampleRate,  samples, Int32(samples.count))
    }
    
    func decode() {
        Decode(recognizer)
    }
    
    func getResult() -> SherpaNcnnRecongitionResult {
        let result:  UnsafeMutablePointer<SherpaNcnnResult>? = GetResult(recognizer)
        return SherpaNcnnRecongitionResult(result: result)
    }
    
    func reset() {
        Reset(recognizer)
    }
    
    func inputFinished() {
        InputFinished(recognizer)
    }
}
