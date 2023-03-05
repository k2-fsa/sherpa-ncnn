package com.k2fsa.sherpa.ncnn;

/**
 * 解码配置
 */
public class DecoderConfig {
    
    public String method = "modified_beam_search"; // 解码方式。固定字段: greedy_search（贪婪搜索）, modified_beam_search（已优化集束搜索）
    public int numActivePaths = 4; // 只作用于modified_beam_search。
    
    public DecoderConfig(String method, int numActivePaths) {
        this.method = method;
        this.numActivePaths = numActivePaths;
    }
    
    public String getMethod() {
        return method;
    }
    
    public void setMethod(String method) {
        this.method = method;
    }
    
    public int getNumActivePaths() {
        return numActivePaths;
    }
    
    public void setNumActivePaths(int numActivePaths) {
        this.numActivePaths = numActivePaths;
    }
}
