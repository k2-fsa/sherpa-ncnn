import AVFoundation

extension AudioBuffer {
  func array() -> [Float] {
    return Array(UnsafeBufferPointer(self))
  }
}

extension AVAudioPCMBuffer {
  func array() -> [Float] {
    return self.audioBufferList.pointee.mBuffers.array()
  }
}

func run() {
  let encoderParam =
    "./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.param"
  let encoderBin =
    "./sherpa-ncnn-conv-emformer-transducer-2022-12-06/encoder_jit_trace-pnnx.ncnn.bin"
  let decoderParam =
    "./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.param"
  let decoderBin =
    "./sherpa-ncnn-conv-emformer-transducer-2022-12-06/decoder_jit_trace-pnnx.ncnn.bin"
  let joinerParam =
    "./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.param"
  let joinerBin = "./sherpa-ncnn-conv-emformer-transducer-2022-12-06/joiner_jit_trace-pnnx.ncnn.bin"
  let tokens = "./sherpa-ncnn-conv-emformer-transducer-2022-12-06/tokens.txt"

  var modelConfig = sherpaNcnnModelConfig(
    encoderParam: encoderParam,
    encoderBin: encoderBin,
    decoderParam: decoderParam,
    decoderBin: decoderBin,
    joinerParam: joinerParam,
    joinerBin: joinerBin,
    tokens: tokens,
    numThreads: 4)

  var decoderConfig = sherpaNcnnDecoderConfig(
    decodingMethod: "modified_beam_search",
    numActivePaths: 4
  )

  let recognizer = SherpaNcnnRecognizer(
    modelConfig: &modelConfig,
    decoderConfig: &decoderConfig)

  let filePath = "./sherpa-ncnn-conv-emformer-transducer-2022-12-06/test_wavs/0.wav"
  let fileURL: NSURL = NSURL(fileURLWithPath: filePath)
  let audioFile = try! AVAudioFile(forReading: fileURL as URL)

  let audioFormat = audioFile.processingFormat
  assert(audioFormat.sampleRate == 16000)
  assert(audioFormat.channelCount == 1)

  let audioFrameCount = UInt32(audioFile.length)
  let audioFileBuffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: audioFrameCount)

  try! audioFile.read(into: audioFileBuffer!)
  let array: [Float]! = audioFileBuffer?.array()
  recognizer.acceptWaveform(samples: array)

  let tailPadding = [Float](repeating: 0.0, count: 3200)
  recognizer.acceptWaveform(samples: tailPadding)

  recognizer.inputFinished()
  recognizer.decode()

  let result = recognizer.getResult()
  print("\nresult is:\n\(result.text)")
}

@main
struct App {
  static func main() {
    run()
  }
}
