package com.k2fsa.sherpa.ncnn

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.provider.MediaStore.Audio
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import org.w3c.dom.Text
import kotlin.concurrent.thread

private const val TAG = "sherpa-ncnn"
private const val REQUEST_RECORD_AUDIO_PERMISSION = 200


class MainActivity : AppCompatActivity() {
    private var permissionToRecordAccepted = false
    private var permissions: Array<String> = arrayOf(Manifest.permission.RECORD_AUDIO)

    private lateinit var model: SherpaNcnn
    private var audioRecord : AudioRecord? = null
    private lateinit var button: Button
    private lateinit var textView: TextView
    private var recordingThread: Thread? = null

    private val audioSource = MediaRecorder.AudioSource.MIC
    private val sampleRateInHz = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    //val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val audioFormat = AudioFormat.ENCODING_PCM_FLOAT
    private var bufferSizeInBytes : Int = 0

    private var isRecording : Boolean = false


    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        permissionToRecordAccepted = if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        } else {
            false
        }

        if (!permissionToRecordAccepted) {
            Log.e(TAG, "Auido record is disallowed")
            finish()
        }

        Log.i(TAG, "Get the permission for audio record")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION)

        Log.i(TAG, "Start to initialize model")
        initModel()
        Log.i(TAG, "Finished initializing model")

        button = findViewById(R.id.my_button)
        button.setOnClickListener {onclick()}

        textView = findViewById(R.id.my_text)
        textView.movementMethod = ScrollingMovementMethod()
    }
    private fun onclick() {
        if(!isRecording) {
            initMicrophone()
            audioRecord!!.startRecording()
            button.text = "Stop"
            isRecording = true
            recordingThread = thread(true) {
                processSamples()
            }
            Log.i(TAG, "Started recording")
        } else {
            isRecording = false
            audioRecord!!.stop()
            audioRecord!!.release()
            audioRecord = null
            Log.i(TAG, "Stopped recording")
        }
    }

    private fun processSamples() {
        Log.i(TAG, "processing samples")
        var buffer = ShortArray(bufferSizeInBytes / 2)
        while(isRecording) {
            val ret = audioRecord!!.read(buffer, 0, buffer.size)
            Log.i(TAG, "Read $ret samples")
        }

    }

    private fun initMicrophone() {
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION)
            return
        }

        bufferSizeInBytes = AudioRecord.getMinBufferSize(sampleRateInHz, channelConfig, audioFormat)
        Log.i(TAG, "buffer size in bytes: $bufferSizeInBytes")

        audioRecord = AudioRecord(
            audioSource,
            sampleRateInHz,
            channelConfig,
            audioFormat,
            bufferSizeInBytes
        )
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
/*
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
 */
    }
}