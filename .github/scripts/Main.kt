package com.k2fsa.sherpa.ncnn

import android.content.res.AssetManager

fun main() {
  var model = SherpaNcnn(
      assetManager = AssetManager(),
      modelConfig = getModelConfig(type = 1, useGPU = false)!!,
      decoderConfig = getDecoderConfig(enableEndpoint = true),
      fbankConfig = getFbankConfig(),
  )

  var samples = WaveReader.readWave(
      assetManager = AssetManager(),
      filename = "./sherpa-ncnn-conv-emformer-transducer-2022-12-06/test_wavs/2.wav"
  )

  model.decodeSamples(samples!!)

  var tail_paddings = FloatArray(8000) // 0.5 seconds
  model.decodeSamples(tail_paddings)

  model.inputFinished()
  println(model.text)

  return


}
