#!/usr/bin/env bash

# This script pack all source code into a single zip file
# so that you can build it locally without accessing Internet
# on your board.

set -e

SHERPA_NCNN_VERSION=$(grep "SHERPA_NCNN_VERSION" ./CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)
echo "SHERPA_NCNN_VERSION: ${SHERPA_NCNN_VERSION}"

dir=sherpa-ncnn-all-in-one-for-embedded-systems-${SHERPA_NCNN_VERSION}

rm -rf $dir
mkdir -p $dir

pushd $dir

wget \
  -O sherpa-ncnn-${SHERPA_NCNN_VERSION}.tar.gz \
  https://github.com/k2-fsa/sherpa-ncnn/archive/refs/tags/v${SHERPA_NCNN_VERSION}.tar.gz

tar xf sherpa-ncnn-${SHERPA_NCNN_VERSION}.tar.gz
rm -v sherpa-ncnn-${SHERPA_NCNN_VERSION}.tar.gz

# Please also change ./build-m3axpi.sh
wget \
  -O kaldi-native-fbank-1.18.5.tar.gz \
  https://github.com/csukuangfj/kaldi-native-fbank/archive/refs/tags/v1.18.5.tar.gz

wget \
  -O ncnn-sherpa-1.1.tar.gz \
  https://github.com/csukuangfj/ncnn/archive/refs/tags/sherpa-1.1.tar.gz

cat >README.md <<EOF
Please put files from this folder to the directory \$HOME/asr/

rm -rf \$HOME/asr
mkdir -p \$HOME/asr

tar xvf sherpa-ncnn-all-in-one-for-embedded-systems-${SHERPA_NCNN_VERSION}.tar.bz2 --strip-components 1 -C \$HOME/asr
rm sherpa-ncnn-all-in-one-for-embedded-systems-${SHERPA_NCNN_VERSION}.tar.bz2  # to save space

ls -lh \$HOME/asr

It should print something like below:

ls -lh \$HOME/asr
total 24368
-rw-r--r--   1 fangjun  staff    59K Feb  2 17:01 kaldi-native-fbank-1.18.5.tar.gz
-rw-r--r--   1 fangjun  staff    12M Feb  2 17:01 sherpa-1.1.tar.gz
drwxr-xr-x  29 fangjun  staff   928B Feb  2 16:05 sherpa-ncnn-${SHERPA_NCNN_VERSION}

# Note: It is OK if the versions of the above files are different.
# The two .tar.gz files must be placed in \$HOME/asr
EOF

popd

tar cjf ${dir}.tar.bz2 $dir
