package com.k2fsa.sherpa.ncnn.wear.os.presentation

import android.content.res.AssetManager
import android.util.Log
import com.k2fsa.sherpa.ncnn.RecognizerConfig
import com.k2fsa.sherpa.ncnn.SherpaNcnn
import com.k2fsa.sherpa.ncnn.getDecoderConfig
import com.k2fsa.sherpa.ncnn.getFeatureExtractorConfig
import com.k2fsa.sherpa.ncnn.getModelConfig

object Recognizer {
    private var _recognizer: SherpaNcnn? = null
    val recognizer: SherpaNcnn
        get() {
            return _recognizer!!
        }
    fun initRecognizer(assetManager: AssetManager? = null) {
        synchronized(this) {
            if(_recognizer != null) {
                return
            }
            Log.i(TAG, "Initializing recognizer")

            val featConfig = getFeatureExtractorConfig(
                sampleRate = 16000.0f,
                featureDim = 80
            )
            val modelConfig = getModelConfig(type = 5, useGPU = false)!!
            val decoderConfig = getDecoderConfig(method = "greedy_search", numActivePaths = 4)
            val config = RecognizerConfig(
                featConfig = featConfig,
                modelConfig = modelConfig,
                decoderConfig = decoderConfig,
                enableEndpoint = true,
                rule1MinTrailingSilence = 2.0f,
                rule2MinTrailingSilence = 0.8f,
                rule3MinUtteranceLength = 20.0f,
            )
            _recognizer = SherpaNcnn(
                assetManager = assetManager,
                config = config,
            )
        }
    }
}