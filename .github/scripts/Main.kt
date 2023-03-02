package com.k2fsa.sherpa.ncnn

import android.content.res.AssetManager

fun main() {
    val featConfig =
        getFeatureExtractorConfig(sampleRate = 16000.0f, featureDim = 80, maxFeatureVectors = -1)
    val modelConfig = getModelConfig(type = 1, useGPU = false)!!
    val decoderConfig = getDecoderConfig(method = "greedy_search", numActivePaths = 4)

    val config = RecognizerConfig(
        featConfig = featConfig,
        modelConfig = modelConfig,
        decoderConfig = decoderConfig,
        enableEndpoint = false,
        rule1MinTrailingSilence = 2.0f,
        rule2MinTrailingSilence = 1.0f,
        rule3MinUtteranceLength = 20.0f,
    )

    var model = SherpaNcnn(
        assetManager = AssetManager(),
        config = config,
    )

    var samples = WaveReader.readWave(
        assetManager = AssetManager(),
        filename = "./sherpa-ncnn-conv-emformer-transducer-2022-12-06/test_wavs/2.wav"
    )
    model.acceptSamples(samples!!)


    var tail_paddings = FloatArray(8000) // 0.5 seconds
    model.acceptSamples(tail_paddings)

    model.inputFinished()
    while (model.isReady()) {
        model.decode()
    }

    println(model.text)

    return
}
