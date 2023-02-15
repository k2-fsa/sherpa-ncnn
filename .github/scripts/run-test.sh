#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

echo "EXE is $EXE"
echo "PATH: $PATH"

which $EXE

log "------------------------------------------------------------"
log "Run ConvEmformer transducer (Chinese, small model)"
log "------------------------------------------------------------"
repo_url=https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-08
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo/v2
git lfs pull --include "encoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.bin"
git lfs pull --include "decoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.bin"
git lfs pull --include "joiner_jit_trace-pnnx-epoch-15-avg-3.ncnn.bin"

git lfs pull --include "encoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.int8.bin"
git lfs pull --include "joiner_jit_trace-pnnx-epoch-15-avg-3.ncnn.int8.bin"

popd

waves=(
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/2.wav
)

for wave in ${waves[@]}; do
  for m in greedy_search modified_beam_search; do
    log "----test $m ---"

    time $EXE \
      $repo/v2/tokens.txt \
      $repo/v2/encoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.param \
      $repo/v2/encoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.bin \
      $repo/v2/decoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.param \
      $repo/v2/decoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.bin \
      $repo/v2/joiner_jit_trace-pnnx-epoch-15-avg-3.ncnn.param \
      $repo/v2/joiner_jit_trace-pnnx-epoch-15-avg-3.ncnn.bin \
      $wave \
      4 \
      $m
  done
done

log "Test int8 models"

for wave in ${waves[@]}; do
  for m in greedy_search modified_beam_search; do
    log "----test $m ---"

    time $EXE \
      $repo/v2/tokens.txt \
      $repo/v2/encoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.int8.param \
      $repo/v2/encoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.int8.bin \
      $repo/v2/decoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.param \
      $repo/v2/decoder_jit_trace-pnnx-epoch-15-avg-3.ncnn.bin \
      $repo/v2/joiner_jit_trace-pnnx-epoch-15-avg-3.ncnn.int8.param \
      $repo/v2/joiner_jit_trace-pnnx-epoch-15-avg-3.ncnn.int8.bin \
      $wave \
      4 \
      $m
  done
done

rm -rf $repo

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
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/2.wav
)

for wave in ${waves[@]}; do
  for m in greedy_search modified_beam_search; do
    log "----test $m ---"

    time $EXE \
      $repo/tokens.txt \
      $repo/encoder_jit_trace-v2-epoch-11-avg-2-pnnx.ncnn.param \
      $repo/encoder_jit_trace-v2-epoch-11-avg-2-pnnx.ncnn.bin \
      $repo/decoder_jit_trace-v2-epoch-11-avg-2-pnnx.ncnn.param \
      $repo/decoder_jit_trace-v2-epoch-11-avg-2-pnnx.ncnn.bin \
      $repo/joiner_jit_trace-v2-epoch-11-avg-2-pnnx.ncnn.param \
      $repo/joiner_jit_trace-v2-epoch-11-avg-2-pnnx.ncnn.bin \
      $wave \
      4 \
      $m
  done
done

rm -rf $repo

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
$repo/test_wavs/1089-134686-0001.wav
$repo/test_wavs/1221-135766-0001.wav
$repo/test_wavs/1221-135766-0002.wav
)

for wave in ${waves[@]}; do
  for m in greedy_search modified_beam_search; do
    log "----test $m ---"

    time $EXE \
      $repo/tokens.txt \
      $repo/bar/encoder_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn.param \
      $repo/bar/encoder_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn.bin \
      $repo/bar/decoder_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn.param \
      $repo/bar/decoder_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn.bin \
      $repo/bar/joiner_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn.param \
      $repo/bar/joiner_jit_trace-v2-iter-468000-avg-16-pnnx.ncnn.bin \
      $wave \
      4 \
      $m
  done
done

rm -rf $repo

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

git lfs pull --include "encoder_jit_trace-epoch-30-avg-10-pnnx.ncnn.int8.bin"
git lfs pull --include "joiner_jit_trace-epoch-30-avg-10-pnnx.ncnn.int8.bin"
popd

waves=(
$repo/test_wavs/1089-134686-0001.wav
$repo/test_wavs/1221-135766-0001.wav
$repo/test_wavs/1221-135766-0002.wav
)

for wave in ${waves[@]}; do
  for m in greedy_search modified_beam_search; do
    log "----test $m ---"

    time $EXE \
      $repo/tokens.txt \
      $repo/encoder_jit_trace-epoch-30-avg-10-pnnx.ncnn.param \
      $repo/encoder_jit_trace-epoch-30-avg-10-pnnx.ncnn.bin \
      $repo/decoder_jit_trace-epoch-30-avg-10-pnnx.ncnn.param \
      $repo/decoder_jit_trace-epoch-30-avg-10-pnnx.ncnn.bin \
      $repo/joiner_jit_trace-epoch-30-avg-10-pnnx.ncnn.param \
      $repo/joiner_jit_trace-epoch-30-avg-10-pnnx.ncnn.bin \
      $wave \
      4 \
      $m
  done
done



log "Test int8 models"

for wave in ${waves[@]}; do
  for m in greedy_search modified_beam_search; do
    log "----test $m ---"

    time $EXE \
      $repo/tokens.txt \
      $repo/encoder_jit_trace-epoch-30-avg-10-pnnx.ncnn.int8.param \
      $repo/encoder_jit_trace-epoch-30-avg-10-pnnx.ncnn.int8.bin \
      $repo/decoder_jit_trace-epoch-30-avg-10-pnnx.ncnn.param \
      $repo/decoder_jit_trace-epoch-30-avg-10-pnnx.ncnn.bin \
      $repo/joiner_jit_trace-epoch-30-avg-10-pnnx.ncnn.int8.param \
      $repo/joiner_jit_trace-epoch-30-avg-10-pnnx.ncnn.int8.bin \
      $wave \
      4 \
      $m
  done
done

rm -rf $repo

log "------------------------------------------------------------"
log "Run ConvEmformer transducer (English + Chinese, mixed model)"
log "------------------------------------------------------------"
repo_url=https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-06
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "encoder_jit_trace-pnnx.ncnn.bin"
git lfs pull --include "decoder_jit_trace-pnnx.ncnn.bin"
git lfs pull --include "joiner_jit_trace-pnnx.ncnn.bin"

# for in8 models
git lfs pull --include "encoder_jit_trace-pnnx.ncnn.int8.bin"
git lfs pull --include "joiner_jit_trace-pnnx.ncnn.int8.bin"
popd
waves=(
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/2.wav
$repo/test_wavs/3.wav
$repo/test_wavs/4.wav
)

for wave in ${waves[@]}; do
  for m in greedy_search modified_beam_search; do
    log "----test $m ---"

    time $EXE \
      $repo/tokens.txt \
      $repo/encoder_jit_trace-pnnx.ncnn.param \
      $repo/encoder_jit_trace-pnnx.ncnn.bin \
      $repo/decoder_jit_trace-pnnx.ncnn.param \
      $repo/decoder_jit_trace-pnnx.ncnn.bin \
      $repo/joiner_jit_trace-pnnx.ncnn.param \
      $repo/joiner_jit_trace-pnnx.ncnn.bin \
      $wave \
      4 \
      $m
  done
done

log "test int8 models"
for wave in ${waves[@]}; do
  for m in greedy_search modified_beam_search; do
    log "----test $m ---"

    time $EXE \
      $repo/tokens.txt \
      $repo/encoder_jit_trace-pnnx.ncnn.int8.param \
      $repo/encoder_jit_trace-pnnx.ncnn.int8.bin \
      $repo/decoder_jit_trace-pnnx.ncnn.param \
      $repo/decoder_jit_trace-pnnx.ncnn.bin \
      $repo/joiner_jit_trace-pnnx.ncnn.int8.param \
      $repo/joiner_jit_trace-pnnx.ncnn.int8.bin \
      $wave \
      4 \
      $m
  done
done

rm -rf $repo

log "------------------------------------------------------------"
log "Run Zipformer transducer (English + Chinese, bilingual)"
log "------------------------------------------------------------"
repo_url=https://huggingface.co/csukuangfj/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.bin"
popd

waves=(
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/2.wav
$repo/test_wavs/3.wav
$repo/test_wavs/4.wav
)

for wave in ${waves[@]}; do
  for m in greedy_search modified_beam_search; do
    log "----test $m ---"

    time $EXE \
      $repo/tokens.txt \
      $repo/encoder_jit_trace-pnnx.ncnn.param \
      $repo/encoder_jit_trace-pnnx.ncnn.bin \
      $repo/decoder_jit_trace-pnnx.ncnn.param \
      $repo/decoder_jit_trace-pnnx.ncnn.bin \
      $repo/joiner_jit_trace-pnnx.ncnn.param \
      $repo/joiner_jit_trace-pnnx.ncnn.bin \
      $wave \
      4 \
      $m
  done
done

rm -rf $repo

log "------------------------------------------------------------"
log "Run Zipformer transducer (English)"
log "------------------------------------------------------------"
repo_url=https://huggingface.co/csukuangfj/sherpa-ncnn-streaming-zipformer-en-2023-02-13
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.bin"
popd

waves=(
$repo/test_wavs/1089-134686-0001.wav
$repo/test_wavs/1221-135766-0001.wav
$repo/test_wavs/1221-135766-0002.wav
)

for wave in ${waves[@]}; do
  for m in greedy_search modified_beam_search; do
    log "----test $m ---"

    time $EXE \
      $repo/tokens.txt \
      $repo/encoder_jit_trace-pnnx.ncnn.param \
      $repo/encoder_jit_trace-pnnx.ncnn.bin \
      $repo/decoder_jit_trace-pnnx.ncnn.param \
      $repo/decoder_jit_trace-pnnx.ncnn.bin \
      $repo/joiner_jit_trace-pnnx.ncnn.param \
      $repo/joiner_jit_trace-pnnx.ncnn.bin \
      $wave \
      4 \
      $m
  done
done

rm -rf $repo

log "------------------------------------------------------------"
log "Run Zipformer transducer (Japanese, fluent)"
log "------------------------------------------------------------"
repo_url=https://huggingface.co/csukuangfj/sherpa-ncnn-streaming-zipformer-ja-fluent-2023-02-14

log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.bin"
git lfs pull --include "test_wavs/*.wav"
popd

waves=(
$repo/test_wavs/aps-smp.wav
$repo/test_wavs/interview_aps-smp.wav
$repo/test_wavs/reproduction-smp.wav
$repo/test_wavs/sps-smp.wav
$repo/test_wavs/task-smp.wav
)

for wave in ${waves[@]}; do
  for m in greedy_search modified_beam_search; do
    log "----test $m ---"

    time $EXE \
      $repo/tokens.txt \
      $repo/encoder_jit_trace-pnnx.ncnn.param \
      $repo/encoder_jit_trace-pnnx.ncnn.bin \
      $repo/decoder_jit_trace-pnnx.ncnn.param \
      $repo/decoder_jit_trace-pnnx.ncnn.bin \
      $repo/joiner_jit_trace-pnnx.ncnn.param \
      $repo/joiner_jit_trace-pnnx.ncnn.bin \
      $wave \
      4 \
      $m
  done
done

rm -rf $repo

log "------------------------------------------------------------"
log "Run Zipformer transducer (Japanese, disfluent)"
log "------------------------------------------------------------"
repo_url=https://huggingface.co/csukuangfj/sherpa-ncnn-streaming-zipformer-ja-disfluent-2023-02-14

log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.bin"
git lfs pull --include "test_wavs/*.wav"
popd

waves=(
$repo/test_wavs/aps-smp.wav
$repo/test_wavs/interview_aps-smp.wav
$repo/test_wavs/reproduction-smp.wav
$repo/test_wavs/sps-smp.wav
$repo/test_wavs/task-smp.wav
)

for wave in ${waves[@]}; do
  for m in greedy_search modified_beam_search; do
    log "----test $m ---"

    time $EXE \
      $repo/tokens.txt \
      $repo/encoder_jit_trace-pnnx.ncnn.param \
      $repo/encoder_jit_trace-pnnx.ncnn.bin \
      $repo/decoder_jit_trace-pnnx.ncnn.param \
      $repo/decoder_jit_trace-pnnx.ncnn.bin \
      $repo/joiner_jit_trace-pnnx.ncnn.param \
      $repo/joiner_jit_trace-pnnx.ncnn.bin \
      $wave \
      4 \
      $m
  done
done

rm -rf $repo
