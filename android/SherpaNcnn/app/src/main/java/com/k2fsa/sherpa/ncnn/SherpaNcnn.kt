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
    var tokens: String,
    var numThreads: Int = 4,
)

class SherpaNcnn(
    assetManager: AssetManager,
    modelConfig: ModelConfig,
    var fbankConfig: FbankOptions,
) {
    private val ptr: Long

    init {
        ptr = new(assetManager, modelConfig, fbankConfig)
    }

    protected fun finalize() {
        delete(ptr)
    }

    fun decodeSamples(samples: FloatArray) =
        decodeSamples(ptr, samples, sampleRate = fbankConfig.frameOpts.sampFreq)

    fun inputFinished() = inputFinished(ptr)
    fun reset() = reset(ptr)

    val text: String
        get() = getText(ptr)

    private external fun new(
        assetManager: AssetManager,
        modelConfig: ModelConfig,
        fbankConfig: FbankOptions
    ): Long

    private external fun delete(ptr: Long)
    private external fun decodeSamples(ptr: Long, samples: FloatArray, sampleRate: Float)
    private external fun inputFinished(ptr: Long)
    private external fun getText(ptr: Long): String
    private external fun reset(ptr: Long)

    companion object {
        init {
            System.loadLibrary("sherpa-ncnn-jni")
        }
    }
}

fun getFbankConfig(): FbankOptions {
    val fbankConfig = FbankOptions()
    fbankConfig.frameOpts.dither = 0.0f
    fbankConfig.melOpts.numBins = 80

    return fbankConfig
}

/*
@param type
0 - https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-04
    This model supports only Chinese

1 - https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-06
    This model supports both English and Chinese

2 - https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-08
    This is a small model with about 18 M parameters. It supports only Chinese
 */
fun getModelConfig(type: Int): ModelConfig? {
    when (type) {
        1 -> {
            val modelDir = "sherpa-ncnn-conv-emformer-transducer-2022-12-06"
            return ModelConfig(
                encoderParam = "$modelDir/encoder_jit_trace-pnnx.ncnn.param",
                encoderBin = "$modelDir/encoder_jit_trace-pnnx.ncnn.bin",
                decoderParam = "$modelDir/decoder_jit_trace-pnnx.ncnn.param",
                decoderBin = "$modelDir/decoder_jit_trace-pnnx.ncnn.bin",
                joinerParam = "$modelDir/joiner_jit_trace-pnnx.ncnn.param",
                joinerBin = "$modelDir/joiner_jit_trace-pnnx.ncnn.bin",
                tokens = "$modelDir/tokens.txt",
                numThreads = 4,
            )

        }
        2 -> {
            val modelDir = "sherpa-ncnn-conv-emformer-transducer-2022-12-08/v2"
            return ModelConfig(
                encoderParam = "$modelDir/encoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.param",
                encoderBin = "$modelDir/encoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.bin",
                decoderParam = "$modelDir/decoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.param",
                decoderBin = "$modelDir/decoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.bin",
                joinerParam = "$modelDir/joiner_jit_trace-pnnx-epoch-15-avg-3.ncnn.param",
                joinerBin = "$modelDir/joiner_jit_trace-pnnx-epoch-15-avg-3.ncnn.bin",
                tokens = "$modelDir/tokens.txt",
                numThreads = 4,
            )
        }
    }
    return null
}