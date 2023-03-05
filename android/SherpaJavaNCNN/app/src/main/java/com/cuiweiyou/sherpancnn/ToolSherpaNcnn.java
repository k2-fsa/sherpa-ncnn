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
    
    // 初始化解码器
    // init SherpaNCNN
    public void init(AssetManager assetManager) {
        FeatureExtractorConfig featConfig = new FeatureExtractorConfig(SAMPLE_RATE_IN_HZ, 80, 1 * 100); // cache 1 second of feature frames
        ModelConfig modelConfig = ModelConfig.getInstance(1, false);   // 声学文件
        DecoderConfig decoderConfig = new DecoderConfig("greedy_search", 4);
        RecognizerConfig config = new RecognizerConfig(featConfig, modelConfig, decoderConfig, true, 2.0f, 0.8f, 20.0f);
    
        sherpaNcnn = new SherpaNcnn(assetManager, config);
        ptr = sherpaNcnn.getPtr();
        Log.e("ard", "翻译id(decoder id)：" + ptr);
    }
    
    // 启动解码线程
    // start decode thread.
    public void start(){
        runnableSherpa = new RunnableSherpa(sherpaNcnn);
        new Thread(runnableSherpa).start();
    }
    
    // 界面ActivityMain2测试用
    // test for ActivityMain2
    public SherpaNcnn getSherpaNcnn(){
        return sherpaNcnn;
    }
    
    // 解码器释放资源
    // SherpaNCNN release
    public void delete() {
        runnableSherpa.stopDecode();
        sherpaNcnn.delete(ptr);
        runnableSherpa = null;
        sherpaNcnn = null;
    }
    
    // 解码
    // SherpaNCNN decode
    public void processSamples(float[] bytes, Handler handler) {
        if (null == bytes || null == handler || null == runnableSherpa) {
            return;
        }
        
        runnableSherpa.decode(bytes, handler);
    }
    
    // 解码线程
    // decode thread
    private class RunnableSherpa implements Runnable {
        private boolean isWorking = true; // 线程存活标记。keep thread alive
        private SherpaNcnn sherpaNcnn;
        private float[] samples;          // audio source
        private Handler handler;          // connect to activity
        
        public RunnableSherpa(SherpaNcnn ncnn) {
            this.sherpaNcnn = ncnn;
        }
        
        // 解码
        // decode
        public void decode(float[] bytes, Handler handler) {
            this.samples = bytes;
            this.handler = handler;
        }
        
        // kill thread
        public void stopDecode() {
            isWorking = false;
        }
        
        @Override
        public void run() {
            while (isWorking) {
                if (null == samples || null == handler) {
                    continue;
                }
                
                sherpaNcnn.acceptWaveform(ptr, samples, SAMPLE_RATE_IN_HZ); // 核心代码。core code.
                while (sherpaNcnn.isReady(ptr)) {
                    sherpaNcnn.decode(ptr);
                }
                
                boolean isEndpoint = sherpaNcnn.isEndpoint(ptr);
                if (isEndpoint) {
                    String text = sherpaNcnn.getText(ptr);
                    Log.e("ard", "-----：" + text);
                    sherpaNcnn.reset(ptr);
                
                    Bundle data = new Bundle();
                    data.putString("text", text);
                    Message msg = handler.obtainMessage();
                    msg.setData(data);
                    handler.sendMessage(msg);
                }
                
                handler = null;
                samples = null;
    
            }
        }
    }
}
