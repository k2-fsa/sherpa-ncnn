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
log "Run Zipformer transducer (Chinese, small model 14M)"
log "------------------------------------------------------------"
repo_url=https://huggingface.co/marcoyang/sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.bin"

waves=(
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/2.wav
)

popd

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
log "Run Zipformer transducer (English, small model 20M)"
log "------------------------------------------------------------"
repo_url=https://huggingface.co/marcoyang/sherpa-ncnn-streaming-zipformer-20M-2023-02-17
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.bin"

waves=(
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/2.wav
)

popd

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
log "Run LSTM transducer (Chinese+English, small model)"
log "------------------------------------------------------------"
repo_url=https://huggingface.co/marcoyang/sherpa-ncnn-lstm-transducer-small-2023-02-13
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "*.bin"

waves=(
$repo/test_wavs/0.wav
$repo/test_wavs/1.wav
$repo/test_wavs/2.wav
)

popd

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

# Decode a URL
if [ $EXE == "sherpa-ncnn-ffmpeg" ]; then
  time $EXE \
    $repo/tokens.txt \
    $repo/encoder_jit_trace-pnnx.ncnn.param \
    $repo/encoder_jit_trace-pnnx.ncnn.bin \
    $repo/decoder_jit_trace-pnnx.ncnn.param \
    $repo/decoder_jit_trace-pnnx.ncnn.bin \
    $repo/joiner_jit_trace-pnnx.ncnn.param \
    $repo/joiner_jit_trace-pnnx.ncnn.bin \
    https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-04/resolve/main/test_wavs/1089-134686-0001.wav \
    4 \
    $m
fi


rm -rf $repo

log "------------------------------------------------------------"
log "Run ConvEmformer transducer (Chinese, small model)"
log "------------------------------------------------------------"
repo_url=https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-08
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "encoder_jit_trace-pnnx.ncnn.bin"
git lfs pull --include "decoder_jit_trace-pnnx.ncnn.bin"
git lfs pull --include "joiner_jit_trace-pnnx.ncnn.bin"

git lfs pull --include "encoder_jit_trace-pnnx.ncnn.int8.bin"
git lfs pull --include "joiner_jit_trace-pnnx.ncnn.int8.bin"

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
      $repo/encoder_jit_trace-pnnx-ncnn.param \
      $repo/encoder_jit_trace-pnnx-ncnn.bin \
      $repo/decoder_jit_trace-pnnx-ncnn.param \
      $repo/decoder_jit_trace-pnnx-ncnn.bin \
      $repo/joiner_jit_trace-pnnx-ncnn.param \
      $repo/joiner_jit_trace-pnnx-ncnn.bin \
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
log "Run LSTM transducer (Chinese)"
log "------------------------------------------------------------"
repo_url=https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-30
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
log "Run LSTM transducer (English)"
log "------------------------------------------------------------"

repo_url=https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-05
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
log "Run ConvEmformer transducer (English)"
log "------------------------------------------------------------"
repo_url=https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-04
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "Download pretrained model and test-data from $repo_url"

GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
pushd $repo
git lfs pull --include "encoder_jit_trace-pnnx.ncnn.bin"
git lfs pull --include "decoder_jit_trace-pnnx.ncnn.bin"
git lfs pull --include "joiner_jit_trace-pnnx.ncnn.bin"

git lfs pull --include "encoder_jit_trace-pnnx.ncnn.int8.bin"
git lfs pull --include "joiner_jit_trace-pnnx.ncnn.int8.bin"
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


log "Test int8 models"

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
log "Run small Zipformer transducer (English + Chinese, bilingual)"
log "------------------------------------------------------------"
repo_url=https://huggingface.co/csukuangfj/sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16
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
