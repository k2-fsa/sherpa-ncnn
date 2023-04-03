## AudioSer介绍

AudioSer是一个先进的深度学习语音识别API服务系统，它可以将上传的.wav格式的语音文件进行转换为文本，并返回给客户端支持多种语言和口音识别，语音转换为文本支持大规模并发请求通过缓存机制避免重复处理相同的文件。

### 技术细节
API使用了sherpa_ncnn库作为深度学习框架，使用了递归神经网络模型和长短时记忆网络模型对声学特征进行建模，对语音信号序列进行处理，实现语音信号的文字转换，我使用了Flask作为 Web 服务框架，通过RESTful API的方式与客户端交互，让其性能发挥最优。

### 目录结构

```python
AudioSer
├───model
│   ├───decoder_jit_trace-pnnx.ncnn.bin
│   ├───...
│   └───tokens.txt
│───cache
│   │───log
│   └───voice
│───sox
│   └───ffmpeg.exe
│───static
│   ├───css
│   ├───...
│   └───src
│───templates
│   └───index.html
└─── AudioSer.py
    |requirements.txt
    │README.md
    |config.py
    └───
```

### 使用说明

安装模块：

```python
pip install -r requirements.txt
```

运行服务:

```python
python AudioSer.py
```

<table style="width:100%">
  <tr>
    <th>AudioSer web</th>

  </tr>
  <tr>
    <td><img src="/python-api-examples/AudioSer/web.png" alt="VITS at training" height="400"></td>

  </tr>
</table>


```python
http://127.0.0.1:5620
```
运行后可以访问WEB界面进行体验测试。

### AP调用

向服务器发送HTTP POST请求，音频以提交字节流方式提交仅支持wav格式。

```pytohn
POST http://127.0.0.1:5620/voice 
Content-Type: audio/wav
file:1.wav
```

### curl

```python
curl -F "file=@E:\Desktop\1.wav" http://127.0.0.1:5620/voice
```

### Python

```python
import requests

url = 'http://127.0.0.1:5620/voice'
file = open('E:/Desktop/1.wav', 'rb')
files = {'file': ('2.wav', file)}
response = requests.post(url, files=files).json()
print(response)
file.close()
```

响应示例：

服务器将返回一段JSON格式的文本。

```json
{ 
    "status": 200, 
    "message": "helloworld"
} 
```

```json
{ 
    "status": 200, 
    "message": "你好世界"
} 
```
