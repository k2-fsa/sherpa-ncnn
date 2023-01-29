#!/usr/bin/env bash

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "PWD is: $PWD"

./build-swift-macos.sh

cd swift-api-examples

log "Download pre-trained models"

git lfs install
git clone https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-06

log "Run ./run-decode-file.sh"

./run-decode-file.sh
