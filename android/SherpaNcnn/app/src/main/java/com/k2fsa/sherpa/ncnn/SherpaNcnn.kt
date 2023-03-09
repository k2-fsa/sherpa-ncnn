package com.k2fsa.sherpa.ncnn

import android.content.res.AssetManager

data class FeatureExtractorConfig(
    var sampleRate: Float,
    var featureDim: Int,
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

data class DecoderConfig(
    var method: String = "modified_beam_search", // valid values: greedy_search, modified_beam_search
    var numActivePaths: Int = 4, // used only by modified_beam_search
)

data class RecognizerConfig(
    var featConfig: FeatureExtractorConfig,
    var modelConfig: ModelConfig,
    var decoderConfig: DecoderConfig,
    var enableEndpoint: Boolean = true,
    var rule1MinTrailingSilence: Float = 2.4f,
    var rule2MinTrailingSilence: Float = 1.0f,
    var rule3MinUtteranceLength: Float = 30.0f,
)

class SherpaNcnn(
    assetManager: AssetManager,
    var config: RecognizerConfig,
) {
    private val ptr: Long

    init {
        ptr = new(assetManager, config)
    }

    protected fun finalize() {
        delete(ptr)
    }

    fun acceptSamples(samples: FloatArray) =
        acceptWaveform(ptr, samples = samples, sampleRate = config.featConfig.sampleRate)

    fun isReady() = isReady(ptr)

    fun decode() = decode(ptr)

    fun inputFinished() = inputFinished(ptr)
    fun isEndpoint(): Boolean = isEndpoint(ptr)
    fun reset() = reset(ptr)

    val text: String
        get() = getText(ptr)

    private external fun new(
        assetManager: AssetManager,
        config: RecognizerConfig,
    ): Long

    private external fun delete(ptr: Long)
    private external fun acceptWaveform(ptr: Long, samples: FloatArray, sampleRate: Float)
    private external fun inputFinished(ptr: Long)
    private external fun isReady(ptr: Long): Boolean
    private external fun decode(ptr: Long): Boolean
    private external fun isEndpoint(ptr: Long): Boolean
    private external fun reset(ptr: Long): Boolean
    private external fun getText(ptr: Long): String

    companion object {
        init {
            System.loadLibrary("sherpa-ncnn-jni")
        }
    }
}

fun getFeatureExtractorConfig(
    sampleRate: Float,
    featureDim: Int
): FeatureExtractorConfig {
    return FeatureExtractorConfig(
        sampleRate = sampleRate,
        featureDim = featureDim,
    )
}

fun getDecoderConfig(method: String, numActivePaths: Int): DecoderConfig {
    return DecoderConfig(method = method, numActivePaths = numActivePaths)
}


/*
@param type
0 - https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-04
    This model supports only Chinese

1 - https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-06
    This model supports both English and Chinese

2 - https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-08
    This is a small model with about 18 M parameters. It supports only Chinese

Please follow
https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
to add more pre-trained models
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
