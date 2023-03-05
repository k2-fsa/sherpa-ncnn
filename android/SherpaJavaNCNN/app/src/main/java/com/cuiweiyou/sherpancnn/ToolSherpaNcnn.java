package com.cuiweiyou.sherpancnn;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;

import com.k2fsa.sherpa.ncnn.DecoderConfig;
import com.k2fsa.sherpa.ncnn.FeatureExtractorConfig;
import com.k2fsa.sherpa.ncnn.ModelConfig;
import com.k2fsa.sherpa.ncnn.RecognizerConfig;
import com.k2fsa.sherpa.ncnn.SherpaNcnn;

/**
 * www.gaohaiyan.com
 */
public class ToolSherpaNcnn {
    public static final int SAMPLE_RATE_IN_HZ = 16000;
    private static ToolSherpaNcnn instance;
    private RunnableSherpa runnableSherpa;
    private SherpaNcnn sherpaNcnn;
    private long ptr;
    
    private ToolSherpaNcnn() {
    }
    
    public static ToolSherpaNcnn getInstance() {
        if (null == instance) {
            synchronized (ToolSherpaNcnn.class) {
                if (null == instance) {
                    instance = new ToolSherpaNcnn();
                }
            }
        }
        return instance;
    }
    
    public void init(AssetManager assetManager) {
        FeatureExtractorConfig featConfig = new FeatureExtractorConfig(SAMPLE_RATE_IN_HZ, 80, 1 * 100); // cache 1 second of feature frames
        ModelConfig modelConfig = ModelConfig.getInstance(1, false);   // 声学文件
        DecoderConfig decoderConfig = new DecoderConfig("greedy_search", 4);
        RecognizerConfig config = new RecognizerConfig(featConfig, modelConfig, decoderConfig, true, 2.0f, 0.8f, 20.0f);
        
        sherpaNcnn = new SherpaNcnn(assetManager, config);
        ptr = sherpaNcnn.getPtr();
        Log.e("ard", "翻译id：" + ptr);
        
        runnableSherpa = new RunnableSherpa(sherpaNcnn);
        new Thread(runnableSherpa).start();
    }
    
    public void delete() {
        runnableSherpa.stopDecode();
        sherpaNcnn.delete(ptr);
        runnableSherpa = null;
        sherpaNcnn = null;
    }
    
    public void processSamples(byte[] bytes, Handler handler) {
        if (null == bytes || null == handler || null == runnableSherpa) {
            return;
        }
        
        runnableSherpa.decode(bytes, handler);
    }
    
    private class RunnableSherpa implements Runnable {
        private boolean isWorking = true;
        private SherpaNcnn sherpaNcnn;
        private byte[] bytes;
        private Handler handler;
        
        public RunnableSherpa(SherpaNcnn ncnn) {
            this.sherpaNcnn = ncnn;
        }
        
        public void decode(byte[] bytes, Handler handler) {
            this.bytes = bytes;
            this.handler = handler;
        }
        
        public void stopDecode() {
            isWorking = false;
        }
        
        @Override
        public void run() {
            while (isWorking) {
                if (null == bytes || null == handler) {
                    continue;
                }
                
                float[] samples = new float[bytes.length];
                for (int i = 0; i < bytes.length; i++) {
                    Float af = Float.valueOf(bytes[i]); // 这里是否有问题？
                    samples[i] = af.floatValue();
                }
                
                sherpaNcnn.acceptWaveform(ptr, samples, sherpaNcnn.recognizerConfig.featConfig.sampleRate);
                while (sherpaNcnn.isReady(ptr)) {
                    sherpaNcnn.decode(ptr);
                }
                
                boolean isEndpoint = sherpaNcnn.isEndpoint(ptr);
                String text = sherpaNcnn.getText(ptr);
                if (isEndpoint) {
                    sherpaNcnn.reset(ptr);
                }
                
                Log.e("ard", "解码：" + text);
                
                Bundle data = new Bundle();
                data.putString("text", text);
                Message msg = handler.obtainMessage();
                msg.setData(data);
                handler.sendMessage(msg);
                
                handler = null;
                bytes = null;
            }
        }
    }
}
