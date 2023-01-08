# Introduction

You can use `sherpa-ncnn` for **real-time** speech recognition (i.e., speech-to-text)
on

  - Linux
  - macOS
  - Windows
  - Embedded Linux (32-bit arm and 64-bit aarch64)
  - Android
  - etc ...

We support all platforms that [ncnn](https://github.com/tencent/ncnn) supports.

Everything can be compiled from source with static link. The generated
executable depends only on system libraries.

**HINT**: It does not depend on PyTorch or any other inference frameworks
other than [ncnn](https://github.com/tencent/ncnn).

Please see the documentation <https://k2-fsa.github.io/sherpa/ncnn/index.html>
for installation and usages, e.g.,

  - How to build an Android app
  - How to download and use pre-trained models

We provide a few YouTube videos for demonstration about real-time speech recognition
with `sherpa-ncnn` using a microphone:

  - `English`: <https://www.bilibili.com/video/BV1TP411p7dh/>
  - `Chinese`: <https://www.bilibili.com/video/BV1214y177vu>

  - Multilingual (Chinese + English) with endpointing Python demo : <https://www.bilibili.com/video/BV1eK411y788/>

  - **Android demos**

  - Multilingual (Chinese + English) Android demo 1: <https://www.bilibili.com/video/BV1Ge411A7XS>
  - Multilingual (Chinese + English) Android demo 2: <https://www.bilibili.com/video/BV1eK411y788/>
  - `Chinese (with background noise)` Android demo : <https://www.bilibili.com/video/BV1GR4y167fx>
  - `Chinese` Android demo : <https://www.bilibili.com/video/BV1744y1Z76H>
  - `Chinese poem with background music` Android demo : <https://www.bilibili.com/video/BV1vR4y1k7eo>


See also <https://github.com/k2-fsa/sherpa>
