#!/usr/bin/env bash
# Copyright (c)  2023  Xiaomi Corporation

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SHERPA_NCNN_DIR=$(cd $SCRIPT_DIR/../.. && pwd)
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "SHERPA_NCNN_DIR: $SHERPA_NCNN_DIR"

SHERPA_NCNN_VERSION=$(grep "SHERPA_NCNN_VERSION" $SHERPA_NCNN_DIR/CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)

# You can pre-download the required wheels to $src_dir

if [ $(hostname) == fangjuns-MacBook-Pro.local ]; then
  HF_MIRROR=hf-mirror.com
  src_dir=/Users/fangjun/open-source/sherpa-ncnn/scripts/dotnet/tmp
else
  src_dir=/tmp
  HF_MIRROR=hf.co
fi
export src_dir

mkdir -p $src_dir
pushd $src_dir

mkdir -p linux-x64 linux-arm64 macos-x64 macos-arm64 windows-x64 windows-x86

linux_x64_wheel_filename=sherpa_ncnn_core-${SHERPA_NCNN_VERSION}-py3-none-manylinux2014_x86_64.whl
linux_x64_wheel=$src_dir/$linux_x64_wheel_filename

linux_arm64_wheel_filename=sherpa_ncnn_core-${SHERPA_NCNN_VERSION}-py3-none-manylinux2014_aarch64.whl
linux_arm64_wheel=$src_dir/$linux_arm64_wheel_filename

macos_x64_wheel_filename=sherpa_ncnn_core-${SHERPA_NCNN_VERSION}-py3-none-macosx_10_15_x86_64.whl
macos_x64_wheel=$src_dir/$macos_x64_wheel_filename

macos_arm64_wheel_filename=sherpa_ncnn_core-${SHERPA_NCNN_VERSION}-py3-none-macosx_11_0_arm64.whl
macos_arm64_wheel=$src_dir/$macos_arm64_wheel_filename

windows_x64_wheel_filename=sherpa_ncnn_core-${SHERPA_NCNN_VERSION}-py3-none-win_amd64.whl
windows_x64_wheel=$src_dir/$windows_x64_wheel_filename

windows_x86_wheel_filename=sherpa_ncnn_core-${SHERPA_NCNN_VERSION}-py3-none-win32.whl
windows_x86_wheel=$src_dir/$windows_x86_wheel_filename

if [ ! -f $src_dir/linux-x64/libsherpa-ncnn-c-api.so ]; then
  echo "---linux x86_64---"
  cd linux-x64
  mkdir -p wheel
  cd wheel
  if [ -f $linux_x64_wheel ]; then
    cp -v $linux_x64_wheel .
  else
    curl -OL https://$HF_MIRROR/csukuangfj/sherpa-ncnn-wheels/resolve/main/$SHERPA_NCNN_VERSION/$linux_x64_wheel_filename
  fi
  unzip $linux_x64_wheel_filename
  cp -v sherpa_ncnn/lib/*.so* ../
  cd ..
  rm -rf wheel
  ls -lh
  cd ..
fi

if [ ! -f $src_dir/linux-arm64/libsherpa-ncnn-c-api.so ]; then
  echo "---linux arm64---"
  cd linux-arm64
  mkdir -p wheel
  cd wheel
  if [ -f $linux_arm64_wheel ]; then
    cp -v $linux_arm64_wheel .
  else
    curl -OL https://$HF_MIRROR/csukuangfj/sherpa-ncnn-wheels/resolve/main/$SHERPA_NCNN_VERSION/$linux_arm64_wheel_filename
  fi
  unzip $linux_arm64_wheel_filename
  cp -v sherpa_ncnn/lib/*.so* ../
  cd ..
  rm -rf wheel
  ls -lh
  cd ..
fi

if [ ! -f $src_dir/macos-x64/libsherpa-ncnn-c-api.dylib ]; then
  echo "--- macOS x86_64---"
  cd macos-x64
  mkdir -p wheel
  cd wheel
  if [ -f $macos_x64_wheel  ]; then
    cp -v $macos_x64_wheel .
  else
    curl -OL https://$HF_MIRROR/csukuangfj/sherpa-ncnn-wheels/resolve/main/$SHERPA_NCNN_VERSION/$macos_x64_wheel_filename
  fi
  unzip $macos_x64_wheel_filename
  cp -v sherpa_ncnn/lib/*.dylib ../

  cd ..

  rm -rf wheel
  ls -lh
  cd ..
fi

if [ ! -f $src_dir/macos-arm64/libsherpa-ncnn-c-api.dylib ]; then
  echo "--- macOS arm64---"
  cd macos-arm64
  mkdir -p wheel
  cd wheel
  if [ -f $macos_arm64_wheel  ]; then
    cp -v $macos_arm64_wheel .
  else
    curl -OL https://$HF_MIRROR/csukuangfj/sherpa-ncnn-wheels/resolve/main/$SHERPA_NCNN_VERSION/$macos_arm64_wheel_filename
  fi
  unzip $macos_arm64_wheel_filename
  cp -v sherpa_ncnn/lib/*.dylib ../

  cd ..

  rm -rf wheel
  ls -lh
  cd ..
fi

if [ ! -f $src_dir/windows-x64/sherpa-ncnn-c-api.dll ]; then
  echo "---windows x64---"
  cd windows-x64
  mkdir -p wheel
  cd wheel
  if [ -f $windows_x64_wheel ]; then
    cp -v $windows_x64_wheel .
  else
    curl -OL https://$HF_MIRROR/csukuangfj/sherpa-ncnn-wheels/resolve/main/$SHERPA_NCNN_VERSION/$windows_x64_wheel_filename
  fi
  unzip $windows_x64_wheel_filename
  cp -v sherpa_ncnn/lib/*.dll ../
  cd ..

  rm -rf wheel
  ls -lh
  cd ..
fi

if [ ! -f $src_dir/windows-x86/sherpa-ncnn-c-api.dll ]; then
  echo "---windows x86---"
  cd windows-x86
  mkdir -p wheel
  cd wheel
  if [ -f $windows_x86_wheel ]; then
    cp -v $windows_x86_wheel .
  else
    curl -OL https://$HF_MIRROR/csukuangfj/sherpa-ncnn-wheels/resolve/main/$SHERPA_NCNN_VERSION/$windows_x86_wheel_filename
  fi
  unzip $windows_x86_wheel_filename
  cp -v sherpa_ncnn/lib/*.dll ../
  cd ..

  rm -rf wheel
  ls -lh
  cd ..
fi

popd

mkdir -p macos-x64 macos-arm64 linux-x64 linux-arm64 windows-x64 windows-x86 all

cp ./sherpa-ncnn.cs all

./generate.py

for d in macos-x64 macos-arm64 linux-x64 linux-arm64 windows-x64 windows-x86 all; do
  pushd $d
  dotnet build -c Release
  dotnet pack -c Release -o ../packages
  popd
done

ls -lh packages

mkdir -p /tmp/packages
cp -v packages/*.nupkg /tmp/packages
