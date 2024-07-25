package com.cuiweiyou.sherpancnn;

import android.media.AudioRecord;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.k2fsa.sherpa.ncnn.SherpaNcnn;

public class ActivityMain2 extends ActivityBase implements View.OnClickListener {
    
    private Button recordButton;
    private ViewWave waveView;
    private TextView textView;
    
    private boolean isRecording = false;
    private AudioRecord audioRecord;
    private SherpaNcnn sherpaNcnn;
    private long ptr;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        textView = findViewById(R.id.my_text);
        waveView = findViewById(R.id.waveView);
        recordButton = findViewById(R.id.record_button);
        recordButton.setOnClickListener(this);
        
        audioRecord = ToolRecorder.getInstance().getAudioRecord();
        sherpaNcnn = ToolSherpaNcnn.getInstance().getSherpaNcnn();
        ptr = sherpaNcnn.getPtr();
        Log.e("ard", "翻译id (decoder id)：" + ptr);
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        isRecording = false;
        audioRecord.stop();
        audioRecord.release();
        sherpaNcnn.delete(ptr);
    }
    
    @Override
    public void onClick(View v) {
        if (!isRecording) {
            isRecording = true;
            audioRecord.startRecording();
            recordButton.setText("点击停止(click to stop)");
            sherpaNcnn.reset(ptr);
            new Thread(new RecordRunnable()).start();
        } else {
            isRecording = false;
            audioRecord.stop();
            recordButton.setText("开始(click to running)");
        }
    }
    
    private class RecordRunnable implements Runnable {
        
        @Override
        public void run() {
            float interval = 0.1f; // i.e., 100 ms
            int bufferSize = (int) (interval * ToolSherpaNcnn.SAMPLE_RATE_IN_HZ); // in samples
            
            while (isRecording) {
                short[] buffer = new short[bufferSize];
                int ret = audioRecord.read(buffer, 0, bufferSize);
                if (ret > 0) {
                    
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            waveView.setWaveform(buffer);
                        }
                    });
                    
                    float[] samples = new float[ret];
                    for (int i = 0; i < ret; i++) {
                        samples[i] = buffer[i] / 32768.0f;
                    }
                    
                    sherpaNcnn.acceptWaveform(ptr, samples, ToolSherpaNcnn.SAMPLE_RATE_IN_HZ);
                    while (sherpaNcnn.isReady(ptr)) {
                        sherpaNcnn.decode(ptr);
                    }
                    
                    boolean isEndpoint = sherpaNcnn.isEndpoint(ptr);
                    if (isEndpoint) {
                        String text = sherpaNcnn.getText(ptr);
                        Log.e("ard", "-----：" + text);
                        sherpaNcnn.reset(ptr);
                        
                        if (!TextUtils.isEmpty(text)) {
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    textView.setText(text);
                                }
                            });
                        }
                    }
                }
            }
        }
    }
}