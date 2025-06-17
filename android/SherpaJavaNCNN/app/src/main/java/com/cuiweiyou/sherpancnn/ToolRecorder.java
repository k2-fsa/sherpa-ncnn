package com.cuiweiyou.sherpancnn;

import android.annotation.SuppressLint;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;

import java.util.Arrays;

/**
 * www.gaohaiyan.com
 */
public class ToolRecorder {
    
    private static ToolRecorder instance;
    private RecordRunner recordRunner;
    private AudioRecord audioRecord;
    private int sampleRateInHz;
    private int minBufferSize;
    
    private ToolRecorder() {
    }
    
    public static ToolRecorder getInstance() {
        if (null == instance) {
            synchronized (ToolRecorder.class) {
                if (null == instance) {
                    instance = new ToolRecorder();
                }
            }
        }
        return instance;
    }
    
    @SuppressLint("MissingPermission")
    public void init() {
        int audioChannel = AudioFormat.CHANNEL_IN_MONO;      // CHANNEL_IN_STEREO双声道，CHANNEL_IN_MONO单声道
        int audioSource = MediaRecorder.AudioSource.MIC;     // 音源
        sampleRateInHz = ToolSherpaNcnn.SAMPLE_RATE_IN_HZ;   // sampleRateInHz 采样率。
        int encodingFormat = AudioFormat.ENCODING_PCM_16BIT; // 采样精度，（ENCODING_PCM_FLOAT须手机api>=23版本才支持-本例不录音）
        minBufferSize = AudioRecord.getMinBufferSize(sampleRateInHz, audioChannel, encodingFormat);
        audioRecord = new AudioRecord(audioSource, sampleRateInHz, audioChannel, encodingFormat, minBufferSize * 2);
    }
    
    // 界面ActivityMain2测试用
    // test for ActivityMain2
    public AudioRecord getAudioRecord() {
        return audioRecord;
    }
    
    // 启动录音
    // start record thread
    public void startRecord(Handler handler1, Handler handler2) {
        recordRunner = new RecordRunner(audioRecord, handler1, handler2);
        new Thread(recordRunner).start();
    }
    
    // 停止录音
    // stop record thread
    public void stopRecord() {
        if (null != recordRunner) {
            recordRunner.stopRecod();
        }
        recordRunner = null;
    }
    
    // release source
    public void release() {
        if (null != recordRunner) {
            recordRunner.releaseRecod();
        }
        recordRunner = null;
    }
    
    class RecordRunner implements Runnable {
        private boolean isRecording = true; // keep thread alive
        private boolean isRelease = false;
        private AudioRecord audioRecord;
        private Handler recordHandler;
        private Handler sherpaHandler;
        
        public RecordRunner(AudioRecord record, Handler handler1, Handler handler2) {
            this.audioRecord = record;
            this.recordHandler = handler1;
            this.sherpaHandler = handler2;
        }
        
        public void stopRecod() {
            this.isRecording = false;
        }
        
        public void releaseRecod() {
            this.isRecording = false;
            this.isRelease = true;
        }
        
        @Override
        public void run() {
            float interval = 0.1f; // i.e., 100 ms
            int bufferSize = (int) (interval * ToolSherpaNcnn.SAMPLE_RATE_IN_HZ); // in samples
            
            audioRecord.startRecording();
            
            while (isRecording) {
                short[] buffer = new short[bufferSize];
                int count = audioRecord.read(buffer, 0, bufferSize);
                if (count > 0) {
                    float[] samples = new float[count];
                    for (int i = 0; i < count; i++) {
                        samples[i] = buffer[i] / 32768.0f;
                    }
                    
                    ToolSherpaNcnn.getInstance().processSamples(samples, sherpaHandler); // 解码。decode
                    
                    short[] tmpbytes = Arrays.copyOf(buffer, count);
                    Bundle data = new Bundle();
                    data.putShortArray("buffer", tmpbytes);
                    Message msg = recordHandler.obtainMessage();
                    msg.setData(data);
                    recordHandler.sendMessage(msg); // 发给ui绘制。send to aty
                }
            }
            
            audioRecord.stop();
            
            if (isRelease) {
                audioRecord.release();
                audioRecord = null;
            }
            recordHandler = null;
            sherpaHandler = null;
        }
    }
}
