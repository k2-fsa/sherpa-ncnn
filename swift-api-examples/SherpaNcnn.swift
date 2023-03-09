/// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
///
/// See LICENSE for clarification regarding multiple authors
///
/// Licensed under the Apache License, Version 2.0 (the "License");
/// you may not use this file except in compliance with the License.
/// You may obtain a copy of the License at
///
///     http://www.apache.org/licenses/LICENSE-2.0
///
/// Unless required by applicable law or agreed to in writing, software
/// distributed under the License is distributed on an "AS IS" BASIS,
/// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
/// See the License for the specific language governing permissions and
/// limitations under the License.

import Foundation  // For NSString

/// Convert a String from swift to a `const char*` so that we can pass it to
/// the C language.
///
/// - Parameters:
///   - s: The String to convert.
/// - Returns: A pointer that can be passed to C as `const char*`

func toCPointer(_ s: String) -> UnsafePointer<Int8>! {
    let cs = (s as NSString).utf8String
    return UnsafePointer<Int8>(cs)
}

/// Return an instance of SherpaNcnnModelConfig.
///
/// Please refer to
/// https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
/// to download the required `.ncnn.param` and `.ncnn.bin` files.
///
/// - Parameters:
///   - encoderParam: Path to encoder.ncnn.param
///   - encoderBin: Path to encoder.ncnn.bin
///   - decoderParam: Path to decoder.ncnn.param
///   - decoderBin: Path to decoder.ncnn.bin
///   - joinerParam: Path to joiner.ncnn.param
///   - joinerBin: Path to joiner.ncnn.bin
///   - tokens.txt: Path to tokens.txt
///   - useVulkanCompute: It if it true, and if sherpa-ncnn is compiled with
///                       vulkan support, and if there are GPUs available, then
///                       it will use GPU for neural network computation.
///                       Otherwise, it uses CPU for computation.
///   - numThreads.txt:  Number of threads to use for neural
///                      network computation.
///
/// - Returns: Return an instance of SherpaNcnnModelConfig
func sherpaNcnnModelConfig(
    encoderParam: String,
    encoderBin: String,
    decoderParam: String,
    decoderBin: String,
    joinerParam: String,
    joinerBin: String,
    tokens: String,
    numThreads: Int = 4,
    useVulkanCompute: Bool = true
) -> SherpaNcnnModelConfig {
    return SherpaNcnnModelConfig(
        encoder_param: toCPointer(encoderParam),
        encoder_bin: toCPointer(encoderBin),
        decoder_param: toCPointer(decoderParam),
        decoder_bin: toCPointer(decoderBin),
        joiner_param: toCPointer(joinerParam),
        joiner_bin: toCPointer(joinerBin),
        tokens: toCPointer(tokens),
        use_vulkan_compute: useVulkanCompute ? 1 : 0,
        num_threads: Int32(numThreads))
}

func sherpaNcnnFeatureExtractorConfig(
    sampleRate: Float,
    featureDim: Int
)-> SherpaNcnnFeatureExtractorConfig {
    return SherpaNcnnFeatureExtractorConfig(
        sampling_rate: sampleRate,
        feature_dim: Int32(featureDim))
}

/// Create an instance of SherpaNcnnDecoderConfig
///
/// - Parameters:
///   - decodingMethod: Valid decoding methods are "greedy_search"
///                     and "modified_beam_search"
///   - numActivePaths: Used only when decodingMethod is "modified_beam_search".
///                     It specifies the beam size for beam search.
///   - enableEndpoint: true to enable endpoint detection. False to disable
///                     endpoint detection.
///   - rule1MinTrailingSilence: An endpoint is detected if trailing silence in
///                              seconds is larger than this value even if
///                              nothing has been decoded. Used only when
///                              enable_endpoint is true.
///   - rule2MinTrailingSilence: An endpoint is detected if trailing silence in
///                              seconds is larger than this value even after
///                              something that is not blank has been decoded.
///                              Used only when enable_endpoint is true.
///   - rule3MinUtteranceLength: An endpoint is detected if the utterance in
///                              seconds is larger than this value.
///                              Used only when enable_endpoint is true.
func sherpaNcnnDecoderConfig(
    decodingMethod: String = "greedy_search",
    numActivePaths: Int = 4
) -> SherpaNcnnDecoderConfig {
    return SherpaNcnnDecoderConfig(
        decoding_method: toCPointer(decodingMethod),
        num_active_paths: Int32(numActivePaths))
}

func sherpaNcnnRecognizerConfig(
    featConfig: SherpaNcnnFeatureExtractorConfig,
    modelConfig: SherpaNcnnModelConfig,
    decoderConfig: SherpaNcnnDecoderConfig,
    enableEndpoint: Bool = false,
    rule1MinTrailingSilence: Float = 2.4,
    rule2MinTrailingSilence: Float = 1.2,
    rule3MinUtteranceLength: Float = 30
) -> SherpaNcnnRecognizerConfig {
    return SherpaNcnnRecognizerConfig(
        feat_config: featConfig,
        model_config: modelConfig,
        decoder_config: decoderConfig,
        enable_endpoint: enableEndpoint ? 1 : 0,
        rule1_min_trailing_silence: rule1MinTrailingSilence,
        rule2_min_trailing_silence: rule2MinTrailingSilence,
        rule3_min_utterance_length: rule3MinUtteranceLength)
}

/// Wrapper for recognition result.
///
/// Usage:
///
///  let result = recognizer.getResult()
///  print("text: \(result.text)")
///
class SherpaNcnnRecongitionResult {
    /// A pointer to the underlying counterpart in C
    let result: UnsafePointer<SherpaNcnnResult>!

    /// Return the actual recognition result.
    /// For English models, it contains words separated by spaces.
    /// For Chinese models, it contains Chinese words.
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
    /// A pointer to the underlying counterpart in C
    let recognizer: OpaquePointer!
    let stream: OpaquePointer!

    /// Constructor taking a model config and a decoder config.
    init(
        config: UnsafePointer<SherpaNcnnRecognizerConfig>!
    ) {
        recognizer = CreateRecognizer(config)
        stream = CreateStream(recognizer)
    }

    deinit {
        if let stream {
            DestroyStream(stream)
        }

        if let recognizer {
            DestroyRecognizer(recognizer)
        }
    }

    /// Decode wave samples.
    ///
    /// - Parameters:
    ///   - samples: Audio samples normalzed to the range [-1, 1]
    ///   - sampleRate: Sample rate of the input audio samples. If it is
    ///                 different from featConfig.sampleRate, we will do
    ///                 resample. Caution: You cannot use a different
    ///                 sampleRate across different calls to
    ///                 AcceptWaveform().
    func acceptWaveform(samples: [Float], sampleRate: Float = 16000) {
        AcceptWaveform(stream, sampleRate, samples, Int32(samples.count))
    }

    func isReady() -> Bool {
        return IsReady(recognizer, stream) == 1 ? true : false
    }

    /// If there are enough number of feature frames, it invokes the neural
    /// network computation and decoding. Otherwise, it is a no-op.
    func decode() {
        Decode(recognizer, stream)
    }

    /// Get the decoding results so far
    func getResult() -> SherpaNcnnRecongitionResult {
        let result: UnsafeMutablePointer<SherpaNcnnResult>? = GetResult(recognizer, stream)
        return SherpaNcnnRecongitionResult(result: result)
    }

    /// Reset the recognizer, which clears the neural network model state
    /// and the state for decoding.
    func reset() {
        Reset(recognizer, stream)
    }

    /// Signal that no more audio samples would be available.
    /// After this call, you cannot call acceptWaveform() any more.
    func inputFinished() {
        InputFinished(stream)
    }

    /// Return true is an endpoint has been detected.
    func isEndpoint() -> Bool {
        return IsEndpoint(recognizer, stream) == 1 ? true : false
    }
}
