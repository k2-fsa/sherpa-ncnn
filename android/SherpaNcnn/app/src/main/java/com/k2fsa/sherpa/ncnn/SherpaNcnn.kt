package com.k2fsa.sherpa.ncnn

import android.content.res.AssetManager

data class FrameExtractionOptions(
    var sampFreq: Float = 16000.0f,
    var frameShiftMs: Float = 10.0f,
    var frameLengthMs: Float = 25.0f,
    var dither: Float = 0.0f,
    var preemphCoeff: Float = 0.97f,
    var removeDcOffset: Boolean = true,
    var windowType: String = "povey",
    var roundToPowerOfTwo: Boolean = true,
    var blackmanCoeff: Float = 0.42f,
    var snipEdges: Boolean = true,
    var maxFeatureVectors: Int = -1
)

data class MelBanksOptions(
    var numBins: Int = 25,
    var lowFreq: Float = 20.0f,
    var highFreq: Float = 0.0f,
    var vtlnLow: Float = 100.0f,
    var vtlnHigh: Float = -500.0f,
    var debugMel: Boolean = false,
    var htkMode: Boolean = false,
)

data class FbankOptions(
    var frameOpts: FrameExtractionOptions = FrameExtractionOptions(),
    var melOpts: MelBanksOptions = MelBanksOptions(),
    var useEnergy: Boolean = false,
    var energyFloor: Float = 0.0f,
    var rawEnergy: Boolean = true,
    var htkCompat: Boolean = false,
    var useLogFbank: Boolean = true,
    var usePower: Boolean = true,
)

data class ModelConfig(
    var encoderParam: String,
    var encoderBin: String,
    var decoderParam: String,
    var decoderBin: String,
    var joinerParam: String,
    var joinerBin: String,
    var numThreads: Int = 4,
)

class SherpaNcnn(
    assetManager: AssetManager,
    var modelConfig: ModelConfig,
    var fbankConfig: FbankOptions,
    tokens: String
) {
    private val ptr: Long

    init {
        ptr = new(assetManager, modelConfig, fbankConfig, tokens)
    }

    protected fun finalize() {
        delete(ptr)
    }

    fun decodeSamples(samples: FloatArray) =
        decodeSamples(ptr, samples, sampleRate = fbankConfig.frameOpts.sampFreq)

    fun inputFinished() = inputFinished(ptr)

    val text: String
        get() = getText(ptr)

    private external fun new(
        assetManager: AssetManager,
        modelConfig: ModelConfig,
        fbankConfig: FbankOptions,
        tokens: String
    ): Long

    private external fun delete(ptr: Long)
    private external fun decodeSamples(ptr: Long, samples: FloatArray, sampleRate: Float)
    private external fun inputFinished(ptr: Long)
    private external fun getText(ptr: Long): String

    companion object {
        init {
            System.loadLibrary("sherpa-ncnn-jni")
        }
    }
}
