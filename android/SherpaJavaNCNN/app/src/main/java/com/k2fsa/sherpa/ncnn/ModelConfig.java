package com.k2fsa.sherpa.ncnn;

/**
 * 声学文件资源
 */
public class ModelConfig {
    public String encoderParam;
    public String encoderBin;
    public String decoderParam;
    public String decoderBin;
    public String joinerParam;
    public String joinerBin;
    public String tokens;
    public int numThreads = 4;
    public boolean useGPU = true; // If there is a GPU and useGPU true, we will use GPU
    
    /**
     * @param encoderParam
     * @param encoderBin
     * @param decoderParam
     * @param decoderBin
     * @param joinerParam
     * @param joinerBin
     * @param tokens
     * @param numThreads
     * @param useGPU
     */
    public ModelConfig(String encoderParam, String encoderBin, String decoderParam, String decoderBin, String joinerParam, String joinerBin, String tokens, int numThreads, boolean useGPU) {
        this.encoderParam = encoderParam;
        this.encoderBin = encoderBin;
        this.decoderParam = decoderParam;
        this.decoderBin = decoderBin;
        this.joinerParam = joinerParam;
        this.joinerBin = joinerBin;
        this.tokens = tokens;
        this.numThreads = numThreads;
        this.useGPU = useGPU;
    }
    
    public String getEncoderParam() {
        return encoderParam;
    }
    
    public void setEncoderParam(String encoderParam) {
        this.encoderParam = encoderParam;
    }
    
    public String getEncoderBin() {
        return encoderBin;
    }
    
    public void setEncoderBin(String encoderBin) {
        this.encoderBin = encoderBin;
    }
    
    public String getDecoderParam() {
        return decoderParam;
    }
    
    public void setDecoderParam(String decoderParam) {
        this.decoderParam = decoderParam;
    }
    
    public String getDecoderBin() {
        return decoderBin;
    }
    
    public void setDecoderBin(String decoderBin) {
        this.decoderBin = decoderBin;
    }
    
    public String getJoinerParam() {
        return joinerParam;
    }
    
    public void setJoinerParam(String joinerParam) {
        this.joinerParam = joinerParam;
    }
    
    public String getJoinerBin() {
        return joinerBin;
    }
    
    public void setJoinerBin(String joinerBin) {
        this.joinerBin = joinerBin;
    }
    
    public String getTokens() {
        return tokens;
    }
    
    public void setTokens(String tokens) {
        this.tokens = tokens;
    }
    
    public int getNumThreads() {
        return numThreads;
    }
    
    public void setNumThreads(int numThreads) {
        this.numThreads = numThreads;
    }
    
    public boolean isUseGPU() {
        return useGPU;
    }
    
    public void setUseGPU(boolean useGPU) {
        this.useGPU = useGPU;
    }
    
    /*
        @param type
        0 - https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-04
            This model supports only Chinese
        1 - https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-06
            This model supports both English and Chinese
        2 - https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-08
            This is a small model with about 18 M parameters. It supports only Chinese
        Please follow
        https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
        to add more pre-trained models.
        If there is a GPU and useGPU is true, we will use GPU
        If there is no GPU and useGPU is true, we won't use GPU
         */
    public static ModelConfig getInstance(int type, boolean useGPU) {
        switch (type) {
            case 1: {
                String modelDir = "transducer";
                return new ModelConfig( //
                                        modelDir + "/encoder_jit_trace-pnnx.ncnn.param", //
                                        modelDir + "/encoder_jit_trace-pnnx.ncnn.bin",   //
                                        modelDir + "/decoder_jit_trace-pnnx.ncnn.param", //
                                        modelDir + "/decoder_jit_trace-pnnx.ncnn.bin",   //
                                        modelDir + "/joiner_jit_trace-pnnx.ncnn.param",  //
                                        modelDir + "/joiner_jit_trace-pnnx.ncnn.bin",    //
                                        modelDir + "/tokens.txt", //
                                        4, //
                                        useGPU);
                
            }
            case 2: {
                String modelDir = "sherpa-ncnn-conv-emformer-transducer-2022-12-08/v2";
                return new ModelConfig( //
                                        modelDir + "/encoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.param", //
                                        modelDir + "/encoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.bin",  //
                                        modelDir + "/decoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.param",  //
                                        modelDir + "/decoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.bin",  //
                                        modelDir + "/joiner_jit_trace-pnnx-epoch-15-avg-3.ncnn.param",  //
                                        modelDir + "/joiner_jit_trace-pnnx-epoch-15-avg-3.ncnn.bin", //
                                        modelDir + "/tokens.txt",  //
                                        4, //
                                        useGPU);
            }
        }
        
        return null;
    }
}
