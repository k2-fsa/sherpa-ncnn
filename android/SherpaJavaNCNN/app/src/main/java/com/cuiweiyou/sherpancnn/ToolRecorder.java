package com.cuiweiyou.sherpancnn;

import android.annotation.SuppressLint;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;

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
    
    public void startRecord(Handler handler1, Handler handler2) {
        recordRunner = new RecordRunner(audioRecord, handler1, handler2);
        new Thread(recordRunner).start();
    }
    
    public void stopRecord() {
        if (null != recordRunner) {
            recordRunner.stopRecod();
        }
        recordRunner = null;
    }
    
    public void release() {
        if (null != recordRunner) {
            recordRunner.releaseRecod();
        }
        recordRunner = null;
    }
    
    class RecordRunner implements Runnable {
        private boolean isRecording = true;
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
            if (null == recordHandler) {
                return;
            }
            if (audioRecord.getState() == AudioRecord.STATE_UNINITIALIZED) {
                return;
            }
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_URGENT_AUDIO);
            
            float interval = 0.1f;                             // i.e., 100 ms
            int bufferSize = (int) (interval * minBufferSize); // in samples
            
            audioRecord.startRecording();
            
            while (isRecording) {
                byte[] buffer = new byte[bufferSize];
                int ret = audioRecord.read(buffer, 0, bufferSize);
                if (ret > 0) {
                    
                    ToolSherpaNcnn.getInstance().processSamples(buffer, sherpaHandler); // 解码
                    
                    byte[] tmpbytes = Arrays.copyOf(buffer, ret);
                    Bundle data = new Bundle();
                    data.putByteArray("buffer", tmpbytes);
                    Message msg = recordHandler.obtainMessage();
                    msg.setData(data);
                    recordHandler.sendMessage(msg); // 发给ui绘制
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
