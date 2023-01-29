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

if [ ! -f openmp-11.0.0.src/build-x86_64/install/include/omp.h ]; then
  pushd openmp-11.0.0.src

  mkdir -p build-x86_64
  cd build-x86_64

  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=install \
    -DCMAKE_OSX_ARCHITECTURES="x86_64" \
    -DLIBOMP_ENABLE_SHARED=OFF \
    -DLIBOMP_OMPT_SUPPORT=OFF \
    -DLIBOMP_USE_HWLOC=OFF ..

  cmake --build . -j 3
  cmake --build . --target install/strip

  popd
fi

export CPLUS_INCLUDE_PATH=$PWD/openmp-11.0.0.src/build-x86_64/install/include:$CPLUS_INCLUDE_PATH
mkdir -p build-x86_64
pushd build-x86_64

cmake \
  -DCMAKE_OSX_ARCHITECTURES="x86_64" \
  -DOpenMP_C_FLAGS="-Xclang -fopenmp" \
  -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
  -DOpenMP_C_LIB_NAMES="libomp" \
  -DOpenMP_CXX_LIB_NAMES="libomp" \
  -DOpenMP_libomp_LIBRARY="$PWD/../openmp-11.0.0.src/build-x86_64/install/lib/libomp.a" \
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

# For openmp.framework
rm -rf openmp.framework
mkdir -p openmp.framework/Versions/A/Headers
mkdir -p openmp.framework/Versions/A/Resources
ln -s A openmp.framework/Versions/Current
ln -s Versions/Current/Headers openmp.framework/Headers
ln -s Versions/Current/Resources openmp.framework/Resources
ln -s Versions/Current/openmp openmp.framework/openmp

cp openmp-11.0.0.src/build-x86_64/install/lib/libomp.a openmp.framework/Versions/A/openmp

cp -a openmp-11.0.0.src/build-x86_64/install/include/* openmp.framework/Versions/A/Headers/
sed -e 's/__NAME__/openmp/g' -e 's/__IDENTIFIER__/org.llvm.openmp/g' -e 's/__VERSION__/11.0/g' ../Info.plist > openmp.framework/Versions/A/Resources/Info.plist

# For sherpa-ncnn.framework
rm -rf sherpa-ncnn.framework
mkdir -p sherpa-ncnn.framework/Versions/A/Headers
mkdir -p sherpa-ncnn.framework/Versions/A/Headers
mkdir -p sherpa-ncnn.framework/Versions/A/Resources
ln -s A sherpa-ncnn.framework/Versions/Current
ln -s Versions/Current/Headers sherpa-ncnn.framework/Headers
ln -s Versions/Current/Resources sherpa-ncnn.framework/Resources
ln -s Versions/Current/sherpa-ncnn sherpa-ncnn.framework/sherpa-ncnn

libtool -static -o sherpa-ncnn.framework/Versions/A/sherpa-ncnn \
  build-x86_64/install/lib/libncnn.a \
  build-x86_64/install/lib/libsherpa-ncnn-c-api.a \
  build-x86_64/install/lib/libsherpa-ncnn-core.a \
  build-x86_64/install/lib/libkaldi-native-fbank-core.a

cp -a build-x86_64/install/include/* sherpa-ncnn.framework/Versions/A/Headers/
sed -e 's/__NAME__/sherpa-ncnn/g' -e 's/__IDENTIFIER__/com.k2-fsa.org/g' -e 's/__VERSION__/1.3.2/g' ../Info.plist > sherpa-ncnn.framework/Versions/A/Resources/Info.plist
