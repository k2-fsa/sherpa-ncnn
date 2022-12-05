#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

echo "EXE is $EXE"

repo_url=https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-30
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "encoder_jit_trace-v2-epoch-11-avg-2-pnnx.ncnn.bin"
git lfs pull --include "decoder_jit_trace-v2-epoch-11-avg-2-pnnx.ncnn.bin"
git lfs pull --include "joiner_jit_trace-v2-epoch-11-avg-2-pnnx.ncnn.bin"
popd

repo_url=https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-05
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "bar/encoder_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn.bin"
git lfs pull --include "bar/decoder_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn.bin"
git lfs pull --include "bar/joiner_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn.bin"
popd

waves=(
./sherpa-ncnn-2022-09-05/test_wavs/1089-134686-0001.wav
./sherpa-ncnn-2022-09-05/test_wavs/1221-135766-0001.wav
./sherpa-ncnn-2022-09-05/test_wavs/1221-135766-0002.wav
)

for wave in ${waves[@]}; do
  time $EXE \
    ./sherpa-ncnn-2022-09-05/tokens.txt \
    ./sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-05/bar/encoder_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-05/bar/decoder_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-05/bar/joiner_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn.bin \
    $wave
done

waves=(
./sherpa-ncnn-2022-09-30/test_wavs/0.wav
./sherpa-ncnn-2022-09-30/test_wavs/1.wav
./sherpa-ncnn-2022-09-30/test_wavs/2.wav
)

for wave in ${waves[@]}; do
  time $EXE \
    ./sherpa-ncnn-2022-09-30/tokens.txt \
    ./sherpa-ncnn-2022-09-30/encoder_jit_trace-v2-epoch-11-avg-2-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-30/encoder_jit_trace-v2-epoch-11-avg-2-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-30/decoder_jit_trace-v2-epoch-11-avg-2-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-30/decoder_jit_trace-v2-epoch-11-avg-2-pnnx.ncnn.bin \
    ./sherpa-ncnn-2022-09-30/joiner_jit_trace-v2-epoch-11-avg-2-pnnx.ncnn.param \
    ./sherpa-ncnn-2022-09-30/joiner_jit_trace-v2-epoch-11-avg-2-pnnx.ncnn.bin \
    $wave
done
