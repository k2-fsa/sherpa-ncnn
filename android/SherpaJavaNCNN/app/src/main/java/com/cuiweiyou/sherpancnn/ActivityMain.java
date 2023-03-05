package com.cuiweiyou.sherpancnn;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.NonNull;

/**
 * www.gaohaiyan.com
 */
public class ActivityMain extends ActivityBase implements View.OnClickListener {
    
    private Button recordButton;
    private ViewWave waveView;
    private TextView textView;
    
    boolean isExit = false;
    private boolean isRecording = false;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        textView = findViewById(R.id.my_text);
        waveView = findViewById(R.id.waveView);
        recordButton = findViewById(R.id.record_button);
        recordButton.setOnClickListener(this);
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        ToolSherpaNcnn.getInstance().delete();
        ToolRecorder.getInstance().release();
    }
    
    @Override
    public void onClick(View v) {
        if (!isRecording) {
            isRecording = true;
            recordButton.setText("点击停止");
            
            RecordHandler handler1 = new RecordHandler(getMainLooper());
            SherpaHandler handler2 = new SherpaHandler(getMainLooper());
            ToolRecorder.getInstance().startRecord(handler1, handler2);
        } else {
            isRecording = false;
            recordButton.setText("点击开始");
            ToolRecorder.getInstance().stopRecord();
        }
    }
    
    private class RecordHandler extends Handler {
        public RecordHandler(@NonNull Looper looper) {
            super(looper);
        }
        
        @Override
        public void handleMessage(@NonNull Message msg) {
            super.handleMessage(msg);
            
            Bundle data = msg.getData();
            byte[] buffer = data.getByteArray("buffer");
            waveView.setWaveform(buffer);
        }
    }
    
    private class SherpaHandler extends Handler {
        public SherpaHandler(@NonNull Looper looper) {
            super(looper);
        }
        
        @Override
        public void handleMessage(@NonNull Message msg) {
            super.handleMessage(msg);
            
            Bundle data = msg.getData();
            String text = data.getString("text");
            
            textView.setText("解码：" + text);
        }
    }
}

