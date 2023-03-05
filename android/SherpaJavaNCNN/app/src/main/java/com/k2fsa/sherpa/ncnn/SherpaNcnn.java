package com.k2fsa.sherpa.ncnn;

import android.content.res.AssetManager;

public class SherpaNcnn {
    
    public RecognizerConfig recognizerConfig;
    private long ptr;
    
    static {
        System.loadLibrary("sherpa-ncnn-jni") ;
        // U AAssetManager_fromJava
        // U AAssetManager_open
        // U AAsset_close
        // U AAsset_getLength
        // U AAsset_read
        // T Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_decodeSamples
        // T Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_delete
        // T Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_getText
        // T Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_inputFinished
        // T Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_newer // 原为new与java关键字重名。
        // T Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_reset
        // T Java_com_k2fsa_sherpa_ncnn_WaveReader_00024Companion_readWave
        //
    }
    
    public SherpaNcnn(AssetManager assetManager, RecognizerConfig config) {
        this.recognizerConfig = config;
        ptr = newer(assetManager, config);
    }
    
    public long getPtr() {
        return ptr;
    }
    
    private native long newer(AssetManager assetManager, RecognizerConfig config); // 初始化，返回解码器id。init, get NCNN id.
    public native void delete(long ptr);
    public native void acceptWaveform(long ptr, float[] samples, float sampleRate); // 开始识别。decode
    public native String getText(long ptr); // 识别结果。decoding result
    public native void inputFinished(long ptr);
    public native boolean isReady(long ptr);
    public native boolean decode(long ptr);
    public native boolean isEndpoint(long ptr);
    public native boolean reset(long ptr);
}