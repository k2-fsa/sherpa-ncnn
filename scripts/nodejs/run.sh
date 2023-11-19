#!/usr/bin/env bash
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SHERPA_NCNN_DIR=$(realpath $SCRIPT_DIR/../..)
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "SHERPA_NCNN_DIR: $SHERPA_NCNN_DIR"

SHERPA_NCNN_VERSION=$(grep "SHERPA_NCNN_VERSION" $SHERPA_NCNN_DIR/CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)

echo "SHERPA_NCNN_VERSION $SHERPA_NCNN_VERSION"
sed -i.bak s/SHERPA_NCNN_VERSION/$SHERPA_NCNN_VERSION/g ./package.json.in
cp package.json.in package.json
rm package.json.in
rm package.json.in.bak

function windows_x64() {
  echo "Process Windows (x64)"
  mkdir -p lib/windows-x64
  dst=$(realpath lib/windows-x64)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-ncnn-wheels/resolve/main/sherpa_ncnn-${SHERPA_NCNN_VERSION}-cp38-cp38-win_amd64.whl
  unzip ./sherpa_ncnn-${SHERPA_NCNN_VERSION}-cp38-cp38-win_amd64.whl
  cp -v sherpa_ncnn/lib/sherpa-ncnn-c-api.dll  $dst/
  cp -v ./*.dll $dst
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
  cp -v ./*.dll $dst
  cd ..
  rm -rf t
}

function linux_x64() {
  echo "Process Linux (x64)"
  mkdir -p lib/linux-x64
  dst=$(realpath lib/linux-x64)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-ncnn-wheels/resolve/main/sherpa_ncnn-${SHERPA_NCNN_VERSION}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  unzip ./sherpa_ncnn-${SHERPA_NCNN_VERSION}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  cp -v sherpa_ncnn.libs/* $dst/
  cp -v sherpa_ncnn/lib/*.so $dst/
  cd ..
  rm -rf t
}

function linux_x86() {
  echo "Process Linux (x86)"
  mkdir -p lib/linux-x86
  dst=$(realpath lib/linux-x86)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-ncnn-wheels/resolve/main/sherpa_ncnn-${SHERPA_NCNN_VERSION}-cp38-cp38-manylinux_2_17_i686.manylinux2014_i686.whl
  unzip ./sherpa_ncnn-${SHERPA_NCNN_VERSION}-cp38-cp38-manylinux_2_17_i686.manylinux2014_i686.whl
  cp -v sherpa_ncnn.libs/* $dst/
  cp -v sherpa_ncnn/lib/*.so $dst/
  cd ..
  rm -rf t
}

function osx_universal2() {
  echo "Process osx-universal2"
  mkdir -p lib/osx-universal2
  dst=$(realpath lib/osx-universal2)
  mkdir t
  cd t
  wget -q https://huggingface.co/csukuangfj/sherpa-ncnn-wheels/resolve/main/sherpa_ncnn-${SHERPA_NCNN_VERSION}-cp38-cp38-macosx_10_9_universal2.whl
  unzip ./sherpa_ncnn-${SHERPA_NCNN_VERSION}-cp38-cp38-macosx_10_9_universal2.whl
  cp -v sherpa_ncnn/lib/*.dylib $dst/
  cd ..
  rm -rf t
}

windows_x64
ls -lh lib/windows-x64

windows_x86
ls -lh lib/windows-x86

linux_x64
ls -lh lib/linux-x64

linux_x86
ls -lh lib/linux-x86

osx_universal2
ls -lh lib/osx-universal2
