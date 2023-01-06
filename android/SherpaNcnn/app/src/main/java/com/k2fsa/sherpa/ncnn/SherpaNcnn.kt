package com.k2fsa.sherpa.ncnn

import android.content.res.AssetManager

data class EndpointRule(
    var mustContainNonSilence: Boolean,
    var minTrailingSilence: Float,
    var minUtteranceLength: Float,
)

data class EndpointConfig(
    var rule1: EndpointRule = EndpointRule(false, 2.4f, 0.0f),
    var rule2: EndpointRule = EndpointRule(true, 1.4f, 0.0f),
    var rule3: EndpointRule = EndpointRule(false, 0.0f, 20.0f)
)

data class DecoderConfig(
    var method: String = "modified_beam_search", // valid values: greedy_search, modified_beam_search
    var numActivePaths: Int = 4, // used only by modified_beam_search
    var useEndpoint: Boolean = true,
    var endpointConfig: EndpointConfig = EndpointConfig(),
)

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
    var useGPU: Boolean = true, // If there is a GPU and useGPU true, we will use GPU
)

class SherpaNcnn(
    assetManager: AssetManager,
    modelConfig: ModelConfig,
    decoderConfig: DecoderConfig,
    var fbankConfig: FbankOptions,
) {
    private val ptr: Long

    init {
        ptr = new(assetManager, modelConfig, decoderConfig, fbankConfig)
    }

    protected fun finalize() {
        delete(ptr)
    }

    fun decodeSamples(samples: FloatArray) =
        decodeSamples(ptr, samples, sampleRate = fbankConfig.frameOpts.sampFreq)

    fun inputFinished() = inputFinished(ptr)
    fun reset() = reset(ptr)
    fun isEndpoint(): Boolean = isEndpoint(ptr)

    val text: String
        get() = getText(ptr)

    private external fun new(
        assetManager: AssetManager,
        modelConfig: ModelConfig,
        decoderConfig: DecoderConfig,
        fbankConfig: FbankOptions
    ): Long

    private external fun delete(ptr: Long)
    private external fun decodeSamples(ptr: Long, samples: FloatArray, sampleRate: Float)
    private external fun inputFinished(ptr: Long)
    private external fun getText(ptr: Long): String
    private external fun reset(ptr: Long)
    private external fun isEndpoint(ptr: Long): Boolean

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
fun getModelConfig(type: Int, useGPU: Boolean): ModelConfig? {
    when (type) {
        1 -> {
            val modelDir = "sherpa-ncnn-conv-emformer-transducer-2022-12-06"
            return ModelConfig(
                encoderParam = "$modelDir/encoder_jit_trace-pnnx.ncnn.int8.param",
                encoderBin = "$modelDir/encoder_jit_trace-pnnx.ncnn.int8.bin",
                decoderParam = "$modelDir/decoder_jit_trace-pnnx.ncnn.param",
                decoderBin = "$modelDir/decoder_jit_trace-pnnx.ncnn.bin",
                joinerParam = "$modelDir/joiner_jit_trace-pnnx.ncnn.int8.param",
                joinerBin = "$modelDir/joiner_jit_trace-pnnx.ncnn.int8.bin",
                tokens = "$modelDir/tokens.txt",
                numThreads = 4,
                useGPU = useGPU,
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
                useGPU = useGPU,
            )
        }
    }
    return null
}

fun getDecoderConfig(useEndpoint: Boolean): DecoderConfig {
    return DecoderConfig(
        method = "modified_beam_search",
        numActivePaths = 4,
        useEndpoint = useEndpoint,
        endpointConfig = EndpointConfig(
            rule1 = EndpointRule(false, 2.4f, 0.0f),
            rule2 = EndpointRule(true, 1.4f, 0.0f),
            rule3 = EndpointRule(false, 0.0f, 20.0f)
        )
    )

}