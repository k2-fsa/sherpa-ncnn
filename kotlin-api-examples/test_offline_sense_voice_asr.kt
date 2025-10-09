package com.k2fsa.sherpa.ncnn

fun main() {
  val recognizer = createOfflineRecognizer(type=1)

  val waveFilename = "./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav"

  val wave = WaveReader.readWave(
      filename = waveFilename,
  )

  val stream = recognizer.createStream()
  stream.acceptWaveform(wave.samples, sampleRate=wave.sampleRate)
  recognizer.decode(stream)

  val result = recognizer.getResult(stream)
  println(result)

  stream.release()
  recognizer.release()
}

fun createOfflineRecognizer(type: Int): OfflineRecognizer {
  val config = OfflineRecognizerConfig(
      featConfig = getFeatureConfig(sampleRate = 16000, featureDim = 80),
      modelConfig = getOfflineModelConfig(type = type)!!,
  )

  return OfflineRecognizer(config = config)
}
