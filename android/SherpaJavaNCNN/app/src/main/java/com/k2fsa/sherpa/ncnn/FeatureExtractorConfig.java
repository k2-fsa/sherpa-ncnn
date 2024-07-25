package com.k2fsa.sherpa.ncnn;

/**
 * 录音配置
 */
public class FeatureExtractorConfig {
    public float sampleRate;
    public int featureDim;        // 特征维度
    public int maxFeatureVectors; // 每个buffer最多能存储帧数
    
    public FeatureExtractorConfig(float sampleRate, int featureDim, int maxFeatureVectors) {
        this.sampleRate = sampleRate;
        this.featureDim = featureDim;
        this.maxFeatureVectors = maxFeatureVectors;
    }
    
    public float getSampleRate() {
        return sampleRate;
    }
    
    public void setSampleRate(float sampleRate) {
        this.sampleRate = sampleRate;
    }
    
    public int getFeatureDim() {
        return featureDim;
    }
    
    public void setFeatureDim(int featureDim) {
        this.featureDim = featureDim;
    }
    
    public int getMaxFeatureVectors() {
        return maxFeatureVectors;
    }
    
    public void setMaxFeatureVectors(int maxFeatureVectors) {
        this.maxFeatureVectors = maxFeatureVectors;
    }
}
