package com.k2fsa.sherpa.ncnn.wear.os.presentation

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.wear.compose.material.Button
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.Text
import com.k2fsa.sherpa.ncnn.wear.os.presentation.theme.SherpaNcnnWearOSTheme
import kotlin.concurrent.thread

private var audioRecord: AudioRecord? = null
private val sampleRateInHz = 16000

@Composable
fun HomeScreen() {
    val activity = LocalContext.current as Activity
    var firstTime by remember { mutableStateOf(true) }
    var isStarted by remember { mutableStateOf(false) }
    var result by remember { mutableStateOf("") }
    val onButtonClick: () -> Unit = {
        firstTime = false;
        isStarted = !isStarted
        if (isStarted) {
            if (ActivityCompat.checkSelfPermission(
                    activity,
                    Manifest.permission.RECORD_AUDIO
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                Log.i(TAG, "Recording is not allowed")
            } else {
                // recording is allowed
                val audioSource = MediaRecorder.AudioSource.MIC
                val channelConfig = AudioFormat.CHANNEL_IN_MONO
                val audioFormat = AudioFormat.ENCODING_PCM_16BIT
                val numBytes =
                    AudioRecord.getMinBufferSize(sampleRateInHz, channelConfig, audioFormat)

                audioRecord = AudioRecord(
                    audioSource,
                    sampleRateInHz,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    numBytes * 2 // a sample has two bytes as we are using 16-bit PCM
                )

                thread(true) {
                    Log.i(TAG, "processing samples")
                    val interval = 0.1 // i.e., 100 ms
                    val bufferSize = (interval * sampleRateInHz).toInt() // in samples
                    val buffer = ShortArray(bufferSize)
                    audioRecord?.let {
                        it.startRecording()
                        while (isStarted) {
                            val ret = audioRecord?.read(buffer, 0, buffer.size)
                            ret?.let { n ->
                                val samples = FloatArray(n) { buffer[it] / 32768.0f }
                                Recognizer.recognizer.acceptSamples(samples)
                                while (Recognizer.recognizer.isReady()) {
                                    Recognizer.recognizer.decode()
                                }
                                val isEndpoint = Recognizer.recognizer.isEndpoint()
                                val text = Recognizer.recognizer.text
                                if (isEndpoint) {
                                    Recognizer.recognizer.reset()
                                }
                                Log.i(TAG, "text: $text")
                                result = text
                            }
                        }
                    }
                    Log.i(TAG, "Stop recording")
                }
            }
        }
    }

    SherpaNcnnWearOSTheme {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colors.background),
            contentAlignment = Alignment.Center
        ) {
            Column(
                // verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Spacer(modifier = Modifier.height(16.dp))

                if (firstTime) {
                    ShowMessage()
                } else {
                    ShowResult(result)
                }
                Spacer(modifier = Modifier.height(32.dp))
                Button(
                    onClick = onButtonClick
                ) {

                    if (isStarted) {
                        Text("Stop")
                    } else {
                        Text("Start")
                    }
                }
                if (!firstTime) {
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        "Next-gen Kaldi + ncnn\nASR",
                        modifier = Modifier.fillMaxWidth(),
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colors.primary,
                        fontSize = 10.sp
                    )
                }
            }
        }
    }
}

@Composable
fun ShowMessage() {
    val msg = "Real-time\nspeech recognition\nwith\nNext-gen Kaldi + ncnn"
    Text(
        modifier = Modifier.fillMaxWidth(),
        textAlign = TextAlign.Center,
        color = MaterialTheme.colors.primary,
        text = msg,
    )
}

@Composable
fun ShowResult(result: String) {
    var msg: String = result
    if (msg.length > 10) {
        val n = 5;
        val first = result.take(n);
        val last = result.takeLast(result.length - n)
        msg = "${first}\n${last}"
    }
    Text(
        modifier = Modifier.fillMaxWidth(),
        textAlign = TextAlign.Center,
        color = MaterialTheme.colors.primary,
        text = msg,
    )
}
