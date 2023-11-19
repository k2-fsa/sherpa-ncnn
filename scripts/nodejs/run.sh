#!/usr/bin/env bash
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SHERPA_NCNN_DIR=$(realpath $SCRIPT_DIR/../..)
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "SHERPA_NCNN_DIR: $SHERPA_NCNN_DIR"

SHERPA_NCNN_VERSION=$(grep "SHERPA_NCNN_VERSION" $SHERPA_NCNN_DIR/CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)

echo "SHERPA_NCNN_VERSION $SHERPA_NCNN_VERSION"

function windows_x64() {
  echo "Process Windows (x64)"
  mkdir -p lib/windows-x64
  dst=$(realpath lib/windows-x64)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-ncnn-wheels/resolve/main/sherpa_ncnn-${SHERPA_NCNN_VERSION}-cp38-cp38-win_amd64.whl
  unzip ./sherpa_ncnn-${SHERPA_NCNN_VERSION}-cp38-cp38-win_amd64.whl
  cp -v sherpa_ncnn/lib/sherpa-ncnn-c-api.dll  $dst/
  cp -v sherpa_ncnn/*.dll $dst
  cd ..
  rm -rf t
}

function windows_x86() {
  echo "Process Windows (x86)"
  mkdir -p lib/windows-x86
  dst=$(realpath lib/windows-x86)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-ncnn-wheels/resolve/main/sherpa_ncnn-${SHERPA_NCNN_VERSION}-cp38-cp38-win32.whl
  unzip ./sherpa_ncnn-${SHERPA_NCNN_VERSION}-cp38-cp38-win32.whl
  cp -v sherpa_ncnn/lib/sherpa-ncnn-c-api.dll  $dst/
  cp -v sherpa_ncnn/*.dll $dst
  cd ..
  rm -rf t
}

windows_x64
ls -lh lib/windows_x64

windows_x86
ls -lh lib/windows_x86
