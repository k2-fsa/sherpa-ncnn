package com.cuiweiyou.sherpancnn;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.AttributeSet;
import android.view.View;

import java.util.Arrays;

/**
 * Created by www.gaohaiyan.com on 2018/4/18.
 */
public class ViewWave extends View {
    
    private byte[] mWaveform;     // 波纹形状
    private Paint paint;
    
    public ViewWave(Context context) {
        super(context);
        init();
    }
    
    public ViewWave(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }
    
    public ViewWave(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }
    
    public void init() {
        int foregroundColour = Color.parseColor("#007aff");
        
        paint = new Paint();
        paint.setColor(foregroundColour);
        paint.setAntiAlias(true);           // 抗锯齿
        paint.setStrokeWidth(1.0f);         // 设置宽度
        paint.setStyle(Paint.Style.STROKE); // 填充
    }
    
    public void setWaveform(byte[] waveform) {
        if (null == waveform) {
            mWaveform = null;
        } else {
            mWaveform = Arrays.copyOf(waveform, waveform.length); // 数组复制
        }
        
        invalidate(); // 设置波纹之后, 需要重绘
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        float width = canvas.getWidth();
        float height = canvas.getHeight();
        
        if (mWaveform != null) {
            renderWaveform(mWaveform, canvas);        // 绘制波形
            renderBlank(width, height, canvas);       // 绘制直线
        } else {
            renderBlank(width, height, canvas);       // 绘制直线
        }
    }
    
    private void renderWaveform(byte[] data, Canvas canvas) {
        if (null == data) {
            return;
        }
        
        float diffX;
        float width = canvas.getWidth() * 1.0f;
        if (data.length > width) {
            diffX = data.length / width;
        } else {
            diffX = width / data.length;
        }
        int height = canvas.getHeight();
        
        Path path = new Path();
        path.moveTo(0, height / 2);
        
        for (int i = 0; i < data.length; i++) {
            float d = data[i];
            float stepX;
            stepX = diffX * i * 1.0f;
            float stepY = d / 2 + (height / 2.0f);
            path.lineTo(stepX, stepY);
        }
        path.moveTo(width, height / 2.0f);
        
        canvas.drawPath(path, paint);
    }
    
    private void renderBlank(float width, float height, Canvas canvas) {
        int y = (int) (height * 0.5f);
        Path path = new Path();
        path.moveTo(0, y);
        path.lineTo(width, y);
        canvas.drawPath(path, paint);
    }
    
    public Handler getWaveHandler() {
        return waveHandler;
    }
    
    private Handler waveHandler = new Handler(Looper.getMainLooper()) {
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            byte[] buffer = (byte[]) msg.obj;
            setWaveform(buffer);
        }
    };
}

