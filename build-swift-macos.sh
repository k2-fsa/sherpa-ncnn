#!/usr/bin/env  bash

set -ex

dir=build-swift-macos
mkdir -p $dir
cd $dir

if [ ! -f openmp-11.0.0.src.tar.xz ]; then
  wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/openmp-11.0.0.src.tar.xz
fi

if [ ! -d openmp-11.0.0.src ]; then
  tar -xf openmp-11.0.0.src.tar.xz
  pushd openmp-11.0.0.src
  sed -i'' -e '/.size __kmp_unnamed_critical_addr/d' runtime/src/z_Linux_asm.S
  sed -i'' -e 's/__kmp_unnamed_critical_addr/___kmp_unnamed_critical_addr/g' runtime/src/z_Linux_asm.S
  popd
fi

if [ ! -f openmp-11.0.0.src/build/install/include/omp.h ]; then
  pushd openmp-11.0.0.src

  mkdir -p build
  cd build

  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=install \
    -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" \
    -DLIBOMP_ENABLE_SHARED=OFF \
    -DLIBOMP_OMPT_SUPPORT=OFF \
    -DLIBOMP_USE_HWLOC=OFF ..

  cmake --build . -j 3
  cmake --build . --target install/strip

  popd
fi

rm -rf  openmp.xcframework

xcodebuild -create-xcframework \
      -library "openmp-11.0.0.src/build/install/lib/libomp.a" \
      -output openmp.xcframework

mkdir -p openmp.xcframework/Headers
cp -v openmp-11.0.0.src/build/install/include/omp.h openmp.xcframework/Headers

export CPLUS_INCLUDE_PATH=$PWD/openmp-11.0.0.src/build/install/include:$CPLUS_INCLUDE_PATH
mkdir -p build
pushd build

cmake \
  -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" \
  -DOpenMP_C_FLAGS="-Xclang -fopenmp" \
  -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
  -DOpenMP_C_LIB_NAMES="libomp" \
  -DOpenMP_CXX_LIB_NAMES="libomp" \
  -DOpenMP_libomp_LIBRARY="$PWD/../openmp-11.0.0.src/build/install/lib/libomp.a" \
  \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_NCNN_ENABLE_PYTHON=OFF \
  -DSHERPA_NCNN_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_NCNN_ENABLE_JNI=OFF \
  -DSHERPA_NCNN_ENABLE_BINARY=OFF \
  -DSHERPA_NCNN_ENABLE_TEST=OFF \
  -DSHERPA_NCNN_ENABLE_C_API=ON \
  ../..

make VERBOSE=1 -j4
make install
rm -rf install/lib/cmake
rm -rf install/lib/pkgconfig
rm -rf install/include/ncnn
rm -rf install/include/kaldi-native-fbank

popd

rm -rf sherpa-ncnn.xcframework

libtool -static -o ./build/install/lib/sherpa-ncnn.a \
  build/install/lib/libncnn.a \
  build/install/lib/libsherpa-ncnn-c-api.a \
  build/install/lib/libsherpa-ncnn-core.a \
  build/install/lib/libkaldi-native-fbank-core.a \
  build/install/lib/libkissfft-float.a

xcodebuild -create-xcframework \
      -library "build/install/lib/sherpa-ncnn.a" \
      -output sherpa-ncnn.xcframework

mkdir -p sherpa-ncnn.xcframework/Headers
cp -av build/install/include/* sherpa-ncnn.xcframework/Headers

pushd sherpa-ncnn.xcframework/macos-arm64_x86_64/
ln -s sherpa-ncnn.a libsherpa-ncnn.a
popd

ls -ld ./*xcframework
