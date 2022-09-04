# Introduction

See <https://github.com/k2-fsa/sherpa>

This repo uses [ncnn](https://github.com/tencent/ncnn) for running the neural
network model and does not depend on libtorch.


# Usage

```bash
git clone https://github.com/k2-fsa/sherpa-ncnn
cd sherpa-ncnn
mkdir build
cd build
cmake ..
make -j

./bin/sherpa-ncnn
```

[ncnn]: https://github.com/tencent/ncnn
