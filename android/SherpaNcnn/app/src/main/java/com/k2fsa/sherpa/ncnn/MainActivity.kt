package com.k2fsa.sherpa.ncnn

import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

private const val TAG = "sherpa-ncnn"

class MainActivity : AppCompatActivity() {
    private lateinit var model: SherpaNcnn
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initModel()
    }

    private fun initModel() {
        // TODO(fangjun): Let the users specify the model directory.
        // If it does not exist, we can download it on the fly.
        val modelDir = "sherpa-ncnn-conv-emformer-transducer-2022-12-06"
        val modelConfig = ModelConfig(
            encoderParam = "$modelDir/encoder_jit_trace-pnnx.ncnn.param",
            encoderBin = "$modelDir/encoder_jit_trace-pnnx.ncnn.bin",
            decoderParam = "$modelDir/decoder_jit_trace-pnnx.ncnn.param",
            decoderBin = "$modelDir/decoder_jit_trace-pnnx.ncnn.bin",
            joinerParam = "$modelDir/joiner_jit_trace-pnnx.ncnn.param",
            joinerBin = "$modelDir/joiner_jit_trace-pnnx.ncnn.bin",
            numThreads = 4,
        )
        val tokens = "$modelDir/tokens.txt"

        val fbankConfig = FbankOptions()
        fbankConfig.frameOpts.dither = 0.0f
        fbankConfig.melOpts.numBins = 80

        val assetManager = application.assets
        model = SherpaNcnn(
            assetManager = assetManager,
            modelConfig = modelConfig,
            fbankConfig = fbankConfig,
            tokens = tokens
        )

        val filename = "$modelDir/test_wavs/0.wav"
        val samples = WaveReader.readWave(assetManager = assetManager, filename = filename)
        if (samples != null) {
            Log.i(TAG, "started")
            model.decodeSamples(samples)
            val tailPaddings = FloatArray(8000) { it * 0F }
            model.decodeSamples(tailPaddings)
            model.inputFinished()
            Log.i(TAG, "done")

            // Get the recognition result
            Log.i(TAG, "text: ${model.text}")

            val view: TextView = findViewById(R.id.my_text)
            view.text = model.text
        }
    }
}