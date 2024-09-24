package com.k2fsa.sherpa.ncnn;

/**
 * 引擎配置
 */
public class RecognizerConfig {
    public FeatureExtractorConfig featConfig; // 录音配置
    public ModelConfig modelConfig;           // 声学模型
    public DecoderConfig decoderConfig;       // 解码配置
    public boolean enableEndpoint = true;         //
    public float rule1MinTrailingSilence = 2.4f;
    public float rule2MinTrailingSilence = 1.0f;
    public float rule3MinUtteranceLength = 30.0f;
    
    public RecognizerConfig(FeatureExtractorConfig featConfig, ModelConfig modelConfig, DecoderConfig decoderConfig, boolean enableEndpoint, float rule1MinTrailingSilence, float rule2MinTrailingSilence, float rule3MinUtteranceLength) {
        this.featConfig = featConfig;
        this.modelConfig = modelConfig;
        this.decoderConfig = decoderConfig;
        this.enableEndpoint = enableEndpoint;
        this.rule1MinTrailingSilence = rule1MinTrailingSilence;
        this.rule2MinTrailingSilence = rule2MinTrailingSilence;
        this.rule3MinUtteranceLength = rule3MinUtteranceLength;
    }
    
    public FeatureExtractorConfig getFeatConfig() {
        return featConfig;
    }
    
    public void setFeatConfig(FeatureExtractorConfig featConfig) {
        this.featConfig = featConfig;
    }
    
    public ModelConfig getModelConfig() {
        return modelConfig;
    }
    
    public void setModelConfig(ModelConfig modelConfig) {
        this.modelConfig = modelConfig;
    }
    
    public DecoderConfig getDecoderConfig() {
        return decoderConfig;
    }
    
    public void setDecoderConfig(DecoderConfig decoderConfig) {
        this.decoderConfig = decoderConfig;
    }
    
    public boolean isEnableEndpoint() {
        return enableEndpoint;
    }
    
    public void setEnableEndpoint(boolean enableEndpoint) {
        this.enableEndpoint = enableEndpoint;
    }
    
    public float getRule1MinTrailingSilence() {
        return rule1MinTrailingSilence;
    }
    
    public void setRule1MinTrailingSilence(float rule1MinTrailingSilence) {
        this.rule1MinTrailingSilence = rule1MinTrailingSilence;
    }
    
    public float getRule2MinTrailingSilence() {
        return rule2MinTrailingSilence;
    }
    
    public void setRule2MinTrailingSilence(float rule2MinTrailingSilence) {
        this.rule2MinTrailingSilence = rule2MinTrailingSilence;
    }
    
    public float getRule3MinUtteranceLength() {
        return rule3MinUtteranceLength;
    }
    
    public void setRule3MinUtteranceLength(float rule3MinUtteranceLength) {
        this.rule3MinUtteranceLength = rule3MinUtteranceLength;
    }
}
