# Introduction

## Usage
```
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13
cd sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13
git lfs pull --include "*.bin"
ls -lh
cd ..

npm install sherpa-ncnn wav
node ./test.js
```
