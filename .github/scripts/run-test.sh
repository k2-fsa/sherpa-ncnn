#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

echo "EXE is $EXE"
echo "PATH: $PATH"

which $EXE

log "------------------------------------------------------------"
log "Run LSTM transducer (Chinese)"
log "------------------------------------------------------------"
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

log "------------------------------------------------------------"
log "Run LSTM transducer (English)"
log "------------------------------------------------------------"

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


log "------------------------------------------------------------"
log "Run ConvEmformer transducer (English)"
log "------------------------------------------------------------"
repo_url=https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-04
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "encoder_jit_trace-epoch-30-avg-10-pnnx.ncnn.bin"
git lfs pull --include "decoder_jit_trace-epoch-30-avg-10-pnnx.ncnn.bin"
git lfs pull --include "joiner_jit_trace-epoch-30-avg-10-pnnx.ncnn.bin"
popd

waves=(
./sherpa-ncnn-conv-emformer-transducer-2022-12-04/test_wavs/1089-134686-0001.wav
./sherpa-ncnn-conv-emformer-transducer-2022-12-04/test_wavs/1221-135766-0001.wav
./sherpa-ncnn-conv-emformer-transducer-2022-12-04/test_wavs/1221-135766-0002.wav
)

for wave in ${waves[@]}; do
  time $EXE \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-04/tokens.txt \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-04/encoder_jit_trace-epoch-30-avg-10-pnnx.ncnn.param \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-04/encoder_jit_trace-epoch-30-avg-10-pnnx.ncnn.bin \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-04/decoder_jit_trace-epoch-30-avg-10-pnnx.ncnn.param \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-04/decoder_jit_trace-epoch-30-avg-10-pnnx.ncnn.bin \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-04/joiner_jit_trace-epoch-30-avg-10-pnnx.ncnn.param \
    ./sherpa-ncnn-conv-emformer-transducer-2022-12-04/joiner_jit_trace-epoch-30-avg-10-pnnx.ncnn.bin \
    $wave
done
