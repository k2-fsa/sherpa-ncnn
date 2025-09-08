### Supported functions

|Real-time Speech recognition|Speech synthesis | Voice activity detection |
|----------------------------|-----------------|--------------------------|
|   ✔️                        |✔️                |         ✔️                |

### Supported platforms

|Architecture| Android          | iOS           | Windows    | macOS | linux |
|------------|------------------|---------------|------------|-------|-------|
|   x64      |  ✔️               |               |   ✔️        | ✔️     |  ✔️    |
|   x86      |  ✔️               |               |   ✔️        |       |       |
|   arm64    |  ✔️               | ✔️             |   ✔️        | ✔️     |  ✔️    |
|   arm32    |  ✔️               |               |            |       |  ✔️    |
|   riscv64  |                  |               |            |       |  ✔️    |

### Supported programming languages

| 1. C++ | 2. C  | 3. Python | 4. JavaScript |
|--------|-------|-----------|---------------|
|   ✔️    | ✔️     | ✔️         |    ✔️          |

|5. Go   | 6. C# | 7. Kotlin | 8. Swift |
|--------|-------|-----------|----------|
| ✔️      |  ✔️    | ✔️         |  ✔️       |


It also supports WebAssembly.

## Introduction

This repository supports running the following functions **locally**

  - Streaming speech-to-text (i.e., real-time speech recognition)
  - Text to speech (e.g., vits models from [piper](https://github.com/OHF-Voice/piper1-gpl))
  - VAD (e.g., [silero-vad](https://github.com/snakers4/silero-vad))

on the following platforms and operating systems:

  - x86, ``x86_64``, 32-bit ARM, 64-bit ARM (arm64, aarch64), RISC-V (riscv64)
  - Linux, macOS, Windows, openKylin
  - Android, WearOS
  - iOS
  - NodeJS
  - WebAssembly
  - [Raspberry Pi](https://www.raspberrypi.com/)
  - [RV1126](https://www.rock-chips.com/uploads/pdf/2022.8.26/191/RV1126%20Brief%20Datasheet.pdf)
  - [LicheePi4A](https://sipeed.com/licheepi4a)
  - [VisionFive 2](https://www.starfivetech.com/en/site/boards)
  - [旭日X3派](https://developer.horizon.ai/api/v1/fileData/documents_pi/index.html)
  - etc

with the following APIs

  - C++, C, Python, Go, ``C#``
  - Kotlin
  - JavaScript
  - Swift

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

### Links for pre-built Android APKs

| Description                    | URL                                                       |
|--------------------------------|-----------------------------------------------------------|
| Streaming speech recognition   | [Address](https://github.com/k2-fsa/sherpa-ncnn/releases) |

### Links for pre-trained models

https://github.com/k2-fsa/sherpa-ncnn/releases/tag/models

### Useful links

- Documentation: https://k2-fsa.github.io/sherpa/ncnn/
- Bilibili 演示视频: https://search.bilibili.com/all?keyword=%E6%96%B0%E4%B8%80%E4%BB%A3Kaldi

### How to reach us

Please see
https://k2-fsa.github.io/sherpa/social-groups.html
for 新一代 Kaldi **微信交流群** and **QQ 交流群**.


## See also

  - <https://github.com/k2-fsa/sherpa-onnx>
  - <https://github.com/k2-fsa/sherpa>
