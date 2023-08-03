#!/usr/bin/env bash

set -ex

git config --global user.email "csukuangfj@gmail.com"
git config --global user.name "Fangjun Kuang"

SHERPA_NCNN_VERSION=v$(grep "SHERPA_NCNN_VERSION" ./CMakeLists.txt  | cut -d " " -f 2  | cut -d '"' -f 2)

echo "========================================================================="

git clone git@github.com:k2-fsa/sherpa-ncnn-go-linux.git

echo "Copy libs for Linux x86_64"

rm -rf sherpa-ncnn-go-linux/lib/x86_64-unknown-linux-gnu/lib*

cp -v ./linux_x86_64/sherpa_ncnn/lib/libkaldi-native-fbank-core.so sherpa-ncnn-go-linux/lib/x86_64-unknown-linux-gnu/
cp -v ./linux_x86_64/sherpa_ncnn/lib/libncnn.so sherpa-ncnn-go-linux/lib/x86_64-unknown-linux-gnu/
cp -v ./linux_x86_64/sherpa_ncnn/lib/libsherpa-ncnn-c-api.so sherpa-ncnn-go-linux/lib/x86_64-unknown-linux-gnu/
cp -v ./linux_x86_64/sherpa_ncnn/lib/libsherpa-ncnn-core.so sherpa-ncnn-go-linux/lib/x86_64-unknown-linux-gnu/
cp -v ./linux_x86_64/sherpa_ncnn.libs/libgomp*.so* sherpa-ncnn-go-linux/lib/x86_64-unknown-linux-gnu/

echo "Copy libs for Linux aarch64"

rm -rf sherpa-ncnn-go-linux/lib/aarch64-unknown-linux-gnu/lib*

cp -v ./linux_aarch64/sherpa_ncnn/lib/libkaldi-native-fbank-core.so sherpa-ncnn-go-linux/lib/aarch64-unknown-linux-gnu/
cp -v ./linux_aarch64/sherpa_ncnn/lib/libncnn.so sherpa-ncnn-go-linux/lib/aarch64-unknown-linux-gnu/
cp -v ./linux_aarch64/sherpa_ncnn/lib/libsherpa-ncnn-c-api.so sherpa-ncnn-go-linux/lib/aarch64-unknown-linux-gnu/
cp -v ./linux_aarch64/sherpa_ncnn/lib/libsherpa-ncnn-core.so sherpa-ncnn-go-linux/lib/aarch64-unknown-linux-gnu/
cp -v ./linux_aarch64/sherpa_ncnn.libs/libgomp*.so* sherpa-ncnn-go-linux/lib/aarch64-unknown-linux-gnu/

echo "Copy sources for Linux"
cp sherpa-ncnn/c-api/c-api.h sherpa-ncnn-go-linux/
cp scripts/go/sherpa_ncnn.go sherpa-ncnn-go-linux/

pushd sherpa-ncnn-go-linux
tag=$(git describe --abbrev=0 --tags)
if [[ x"$VERSION" == x"auto" ]]; then
  # this is a pre-release
  if [[ $tag == ${SHERPA_NCNN_VERSION}* ]]; then
    # echo we have already release pre-release before, so just increment it
    last=$(echo $tag | rev | cut -d'.' -f 1 | rev)
    new_last=$((last+1))
    new_tag=${SHERPA_NCNN_VERSION}-alpha.${new_last}
  else
    new_tag=${SHERPA_NCNN_VERSION}-alpha.1
  fi
else
  new_tag=$VERSION
fi

echo "new_tag: $new_tag"
git add .
git status
git commit -m "Release $new_tag" && \
git push && \
git tag $new_tag && \
git push origin $new_tag || true

popd
echo "========================================================================="

git clone git@github.com:k2-fsa/sherpa-ncnn-go-macos.git

echo "Copy libs for macOS x86_64"
rm -rf sherpa-ncnn-go-macos/lib/x86_64-apple-darwin/lib*
cp -v ./macos-x86_64/libkaldi-native-fbank-core.dylib sherpa-ncnn-go-macos/lib/x86_64-apple-darwin
cp -v ./macos-x86_64/libncnn.dylib sherpa-ncnn-go-macos/lib/x86_64-apple-darwin
cp -v ./macos-x86_64/libsherpa-ncnn-c-api.dylib sherpa-ncnn-go-macos/lib/x86_64-apple-darwin
cp -v ./macos-x86_64/libsherpa-ncnn-core.dylib sherpa-ncnn-go-macos/lib/x86_64-apple-darwin

echo "Copy libs for macOS arm64"
rm -rf sherpa-ncnn-go-macos/lib/aarch64-apple-darwin/lib*
cp -v ./macos-arm64/libkaldi-native-fbank-core.dylib sherpa-ncnn-go-macos/lib/aarch64-apple-darwin
cp -v ./macos-arm64/libncnn.dylib sherpa-ncnn-go-macos/lib/aarch64-apple-darwin
cp -v ./macos-arm64/libsherpa-ncnn-c-api.dylib sherpa-ncnn-go-macos/lib/aarch64-apple-darwin
cp -v ./macos-arm64/libsherpa-ncnn-core.dylib sherpa-ncnn-go-macos/lib/aarch64-apple-darwin

echo "Copy sources for macOS"
cp sherpa-ncnn/c-api/c-api.h sherpa-ncnn-go-macos/
cp scripts/go/sherpa_ncnn.go sherpa-ncnn-go-macos/

pushd sherpa-ncnn-go-macos
tag=$(git describe --abbrev=0 --tags)
if [[ x"$VERSION" == x"auto" ]]; then
  # this is a pre-release
  if [[ $tag == ${SHERPA_NCNN_VERSION}* ]]; then
    # echo we have already release pre-release before, so just increment it
    last=$(echo $tag | rev | cut -d'.' -f 1 | rev)
    new_last=$((last+1))
    new_tag=${SHERPA_NCNN_VERSION}-alpha.${new_last}
  else
    new_tag=${SHERPA_NCNN_VERSION}-alpha.1
  fi
else
  new_tag=$VERSION
fi

echo "new_tag: $new_tag"
git add .
git status
git commit -m "Release $new_tag" && \
git push && \
git tag $new_tag && \
git push origin $new_tag || true

popd
echo "========================================================================="

git clone git@github.com:k2-fsa/sherpa-ncnn-go-windows.git
echo "Copy libs for Windows x86_64"
rm -fv sherpa-ncnn-go-windows/lib/x86_64-pc-windows-gnu/*
cp -v ./windows-x64/kaldi-native-fbank-core.dll sherpa-ncnn-go-windows/lib/x86_64-pc-windows-gnu
cp -v ./windows-x64/ncnn.dll sherpa-ncnn-go-windows/lib/x86_64-pc-windows-gnu
cp -v ./windows-x64/sherpa-ncnn-c-api.dll sherpa-ncnn-go-windows/lib/x86_64-pc-windows-gnu
cp -v ./windows-x64/sherpa-ncnn-core.dll sherpa-ncnn-go-windows/lib/x86_64-pc-windows-gnu

echo "Copy libs for Windows x86"
rm -fv sherpa-ncnn-go-windows/lib/i686-pc-windows-gnu/*
cp -v ./windows-win32/kaldi-native-fbank-core.dll sherpa-ncnn-go-windows/lib/i686-pc-windows-gnu
cp -v ./windows-win32/ncnn.dll sherpa-ncnn-go-windows/lib/i686-pc-windows-gnu
cp -v ./windows-win32/sherpa-ncnn-c-api.dll sherpa-ncnn-go-windows/lib/i686-pc-windows-gnu
cp -v ./windows-win32/sherpa-ncnn-core.dll sherpa-ncnn-go-windows/lib/i686-pc-windows-gnu

echo "Copy sources for Windows"
cp sherpa-ncnn/c-api/c-api.h sherpa-ncnn-go-windows/
cp scripts/go/sherpa_ncnn.go sherpa-ncnn-go-windows/

pushd sherpa-ncnn-go-windows
tag=$(git describe --abbrev=0 --tags)
if [[ x"$VERSION" == x"auto" ]]; then
  # this is a pre-release
  if [[ $tag == ${SHERPA_NCNN_VERSION}* ]]; then
    # echo we have already release pre-release before, so just increment it
    last=$(echo $tag | rev | cut -d'.' -f 1 | rev)
    new_last=$((last+1))
    new_tag=${SHERPA_NCNN_VERSION}-alpha.${new_last}
  else
    new_tag=${SHERPA_NCNN_VERSION}-alpha.1
  fi
else
  new_tag=$VERSION
fi

echo "new_tag: $new_tag"
git add .
git status
git commit -m "Release $new_tag" && \
git push && \
git tag $new_tag && \
git push origin $new_tag || true

popd

echo "========================================================================="


rm -fv ~/.ssh/github
