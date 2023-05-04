#!/usr/bin/env bash

set -ex

if [ -z $API_KEY ]; then
  echo "Please set API_KEY first"
  exit 1
fi

# rm -rf macos linux windows all
mkdir -p macos linux windows all

cp ./sherpa-ncnn.cs all

./generate.py

pushd linux
dotnet build -c Release
dotnet pack -c Release -o ../packages
popd

pushd macos
dotnet build -c Release
dotnet pack -c Release -o ../packages
popd

pushd windows
dotnet build -c Release
dotnet pack -c Release -o ../packages
popd


pushd all
dotnet build -c Release
dotnet pack -c Release -o ../packages
popd

ls -lh packages
