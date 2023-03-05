<br/>
<img src="app/src/main/res/mipmap-xxhdpi/ncnn.png" width="128"/>
<br/>

sherpa-ncnn/android/SherpaNcnn 的 java 版
======
sherpa-ncnn/android/SherpaNcnn, code for Java. <br/>
<table>
    <tr>
        <td>模型及解码工具(pre-trained models & util )</td>
        <td>src/main/java/com/cuiweiyou/sherpancnn/ToolSherpaNcnn.java</td> </tr>
    <tr>
        <td>录音工具(record util)</td>
        <td>src/main/java/com/cuiweiyou/sherpancnn/ToolRecorder.java</td>
    </tr>
    <tr>
        <td>权限请求及工具初始化(request premission, init utils)</td>
        <td>src/main/java/com/cuiweiyou/sherpancnn/ActivityFlash.java</td>
    </tr>
    <tr>
        <td>主界面/操作界面(Main ui)</td>
        <td>src/main/java/com/cuiweiyou/sherpancnn/ActivityMain.java</td>
    </tr>
    <tr>
        <td>直改原Kotlin界面(replace original kotlin code)</td>
        <td>src/main/java/com/cuiweiyou/sherpancnn/ActivityMain2.java</td>
    </tr>
</table>
<br/>　　
<br/>

注意 Warning
======
本例依 sherpa-ncnn/releases/tag/v1.6.0 测试。<br/>
(This example is tested according to sherpa-ncnn/releases/tag/v1.6.0. ) <br/>
在java中"new"是关键字，所以本示例的so是将 sherpa-ncnn-1.6.0/sherpa-ncnn/jni/jni.cc 约245行的函数 Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_new 改为Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_newer 后重新编译。<br/>
(In Java, "new" is the keyword, so in this example, Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_new is replaced by Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_newer in sherpa-ncnn-1.6.0/sherpa-ncnn/jni/jni.cc  about line 245.)<br/>
编译方法参考：(how to build : )<br/>
https://k2-fsa.github.io/sherpa/ncnn/android/build-sherpa-ncnn.html#build-sherpa-ncnn-c-code<br/>
