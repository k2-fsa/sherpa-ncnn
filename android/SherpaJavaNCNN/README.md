      
       
<br/>       
        
sherpa-ncnn/android/SherpaNcnn 的 java 版
====
<img src="app/src/main/res/mipmap-xxhdpi/ncnn.png" width="128"/><br/>
<br/>
模型及解码工具 src/main/java/com/cuiweiyou/sherpancnn/ToolSherpaNcnn.java <br/>
录音工具      src/main/java/com/cuiweiyou/sherpancnn/ToolRecorder.java   <br/>
权限请求及工具初始化 src/main/java/com/cuiweiyou/sherpancnn/ActivityFlash.java <br/>
主界面/操作界面     src/main/java/com/cuiweiyou/sherpancnn/ActivityMain.java  <br/>
<br/>　　
<br/>

说明
====
本例依 sherpa-ncnn/releases/tag/v1.6.0 测试。<br/>
在java中"new"是关键字，所以本示例的so是将 sherpa-ncnn-1.6.0/sherpa-ncnn/jni/jni.cc 约245行的函数 Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_new 改为Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_newer 后重新编译。
