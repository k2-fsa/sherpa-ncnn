package com.k2fsa.sherpa.ncnn

import android.content.res.AssetManager

data class OfflineRecognizerResult(
    val text: String,
    val tokens: Array<String>,
    val timestamps: FloatArray,

    // valid only for sense voice models
    val lang: String,

    // valid only for sense voice models
    val emotion: String,

    // valid only for sense voice models
    val event: String,
)

data class OfflineSenseVoiceModelConfig(
    var modelDir: String = "",
    var language: String = "auto",
    var useInverseTextNormalization: Boolean = true,
)

data class OfflineModelConfig(
    var senseVoice: OfflineSenseVoiceModelConfig = OfflineSenseVoiceModelConfig(),
    var numThreads: Int = 1,
    var debug: Boolean = false,
    var tokens: String = "",
)

data class OfflineRecognizerConfig(
    var featConfig: FeatureConfig = FeatureConfig(),
    var modelConfig: OfflineModelConfig = OfflineModelConfig(),
    var decodingMethod: String = "greedy_search",
    var blankPenalty: Float = 0.0f,
)

class OfflineRecognizer(
    // Please set assetManager to null if you want to load model files from SD card.
    // In that case, ALL PATHS should use absolute paths, i.e., start with /
    assetManager: AssetManager? = null,
    val config: OfflineRecognizerConfig,
) {
    private var ptr: Long

    init {
        ptr = if (assetManager != null) {
            newFromAsset(assetManager, config)
        } else {
            newFromFile(config)
        }
    }

    protected fun finalize() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0
        }
    }

    fun release() = finalize()

    fun createStream(): OfflineStream {
        val p = createStream(ptr)
        return OfflineStream(p)
    }

    fun getResult(stream: OfflineStream): OfflineRecognizerResult {
        val objArray = getResult(stream.ptr)

        val text = objArray[0] as String
        val tokens = objArray[1] as Array<String>
        val timestamps = objArray[2] as FloatArray
        val lang = objArray[3] as String
        val emotion = objArray[4] as String
        val event = objArray[5] as String
        return OfflineRecognizerResult(
            text = text,
            tokens = tokens,
            timestamps = timestamps,
            lang = lang,
            emotion = emotion,
            event = event,
        )
    }

    fun decode(stream: OfflineStream) = decode(ptr, stream.ptr)

    fun setConfig(config: OfflineRecognizerConfig) = setConfig(ptr, config)

    private external fun delete(ptr: Long)

    private external fun createStream(ptr: Long): Long

    private external fun setConfig(ptr: Long, config: OfflineRecognizerConfig)

    private external fun newFromAsset(
        assetManager: AssetManager,
        config: OfflineRecognizerConfig,
    ): Long

    private external fun newFromFile(
        config: OfflineRecognizerConfig,
    ): Long

    private external fun decode(ptr: Long, streamPtr: Long)

    private external fun getResult(streamPtr: Long): Array<Any>

    companion object {
        init {
            System.loadLibrary("sherpa-ncnn-jni")
        }
    }
}

/*
Please see
https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
for a list of pre-trained models.

We only add a few here. Please change the following code
to add your own. (It should be straightforward to add a new model
by following the code)

@param type

0 - sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17 (Chinese, English, Japanese, Korean, Cantonese, 中英日韩粤语)
    https://github.com/k2-fsa/sherpa-ncnn/releases/download/asr-models/sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
    See also its fp16 models at https://k2-fsa.github.io/sherpa/ncnn/sense-voice/pretrained.html#sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17-chinese-english-japanese-korean-cantonese

 */
fun getOfflineModelConfig(type: Int): OfflineModelConfig? {
    when (type) {
        0 -> {
            // int8
            val modelDir = "sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
            return OfflineModelConfig(
                senseVoice = OfflineSenseVoiceModelConfig(
                    modelDir = modelDir,
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        1 -> {
            // fp16
            val modelDir = "sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17"
            return OfflineModelConfig(
                senseVoice = OfflineSenseVoiceModelConfig(
                    modelDir = modelDir,
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }
    }
    return null
}
