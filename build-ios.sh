#!/usr/bin/env  bash

set -ex

dir=build-ios
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

if [ ! -f openmp-11.0.0.src/build/os64/install/include/omp.h ]; then
  pushd openmp-11.0.0.src

  mkdir -p build

  # iOS & simulator running on arm64 & x86_64
  cmake -S . \
    -DCMAKE_TOOLCHAIN_FILE=../../toolchains/ios.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=install \
    -DPLATFORM=OS64 \
    -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 \
    -DPERL_EXECUTABLE=$(which perl) \
    -DLIBOMP_ENABLE_SHARED=OFF \
    -DLIBOMP_OMPT_SUPPORT=OFF \
    -DLIBOMP_USE_HWLOC=OFF \
    -B build/os64

  cmake -S . \
    -DCMAKE_TOOLCHAIN_FILE=../../toolchains/ios.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=install \
    -DPLATFORM=SIMULATORARM64 \
    -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 \
    -DPERL_EXECUTABLE=$(which perl) \
    -DLIBOMP_ENABLE_SHARED=OFF \
    -DLIBOMP_OMPT_SUPPORT=OFF \
    -DLIBOMP_USE_HWLOC=OFF \
    -B build/simulator_arm64

  cmake -S . \
    -DCMAKE_TOOLCHAIN_FILE=../../toolchains/ios.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=install \
    -DPLATFORM=SIMULATOR64 \
    -DENABLE_BITCODE=0 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 \
    -DPERL_EXECUTABLE=$(which perl) \
    -DLIBOMP_ENABLE_SHARED=OFF \
    -DLIBOMP_OMPT_SUPPORT=OFF \
    -DLIBOMP_USE_HWLOC=OFF \
    -B build/simulator_x86_64

  # It generates the following files in the directory install
  # .
  # ├── include
  # │   └── omp.h
  # └── lib
  #     ├── libgomp.a -> libomp.a
  #     ├── libiomp5.a -> libomp.a
  #     └── libomp.a
  #
  # 2 directories, 4 files
  cmake --build ./build/os64 -j 4
  # Generate header for sharper-ncnn.xcframework
  cmake --build ./build/os64 --target install
  cmake --build ./build/simulator_arm64 -j 4
  cmake --build ./build/simulator_x86_64 -j 4

  mkdir -p "./build/simulator/openmp"
  lipo -create build/simulator_x86_64/runtime/src/libomp.a \
               build/simulator_arm64/runtime/src/libomp.a \
       -output build/simulator/openmp/libomp.a

  # Return to parent directory to create xcframework
  popd

  rm -rf  openmp.xcframework 
  xcodebuild -create-xcframework \
        -library "openmp-11.0.0.src/build/os64/runtime/src/libomp.a" \
        -library "openmp-11.0.0.src/build/simulator/openmp/libomp.a" \
        -output openmp.xcframework
  # Copy Headers
  mkdir -p openmp.xcframework/Headers
  cp -v openmp-11.0.0.src/install/include/omp.h openmp.xcframework/Headers
fi


export CPLUS_INCLUDE_PATH=$PWD/openmp.xcframework/Headers/:$CPLUS_INCLUDE_PATH
mkdir -p build

cmake -S .. \
  -DCMAKE_TOOLCHAIN_FILE=./toolchains/ios.toolchain.cmake \
  -DPLATFORM=OS64 \
  -DENABLE_BITCODE=0 \
  -DENABLE_ARC=0 \
  -DENABLE_VISIBILITY=0 \
  -DOpenMP_C_FLAGS="-Xclang -fopenmp" \
  -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
  -DOpenMP_C_LIB_NAMES="libomp" \
  -DOpenMP_CXX_LIB_NAMES="libomp" \
  -DOpenMP_libomp_LIBRARY="$PWD/openmp.xcframework/ios-arm64/libomp.a" \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_NCNN_ENABLE_PYTHON=OFF \
  -DSHERPA_NCNN_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_NCNN_ENABLE_JNI=OFF \
  -DSHERPA_NCNN_ENABLE_BINARY=OFF \
  -DSHERPA_NCNN_ENABLE_TEST=OFF \
  -DSHERPA_NCNN_ENABLE_C_API=ON \
  -B build/os64 

cmake -S .. \
  -DCMAKE_TOOLCHAIN_FILE=./toolchains/ios.toolchain.cmake \
  -DPLATFORM=SIMULATORARM64 \
  -DENABLE_BITCODE=0 \
  -DENABLE_ARC=0 \
  -DENABLE_VISIBILITY=0 \
  -DOpenMP_C_FLAGS="-Xclang -fopenmp" \
  -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
  -DOpenMP_C_LIB_NAMES="libomp" \
  -DOpenMP_CXX_LIB_NAMES="libomp" \
  -DOpenMP_libomp_LIBRARY="$PWD/openmp.xcframework/ios-arm64_x86_64-simulator/libomp.a" \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_NCNN_ENABLE_PYTHON=OFF \
  -DSHERPA_NCNN_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_NCNN_ENABLE_JNI=OFF \
  -DSHERPA_NCNN_ENABLE_BINARY=OFF \
  -DSHERPA_NCNN_ENABLE_TEST=OFF \
  -DSHERPA_NCNN_ENABLE_C_API=ON \
  -B build/simulator_arm64

cmake -S .. \
  -DCMAKE_TOOLCHAIN_FILE=./toolchains/ios.toolchain.cmake \
  -DPLATFORM=SIMULATOR64 \
  -DENABLE_BITCODE=0 \
  -DENABLE_ARC=0 \
  -DENABLE_VISIBILITY=0 \
  -DOpenMP_C_FLAGS="-Xclang -fopenmp" \
  -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
  -DOpenMP_C_LIB_NAMES="libomp" \
  -DOpenMP_CXX_LIB_NAMES="libomp" \
  -DOpenMP_libomp_LIBRARY="$PWD/openmp.xcframework/ios-arm64_x86_64-simulator/libomp.a" \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_NCNN_ENABLE_PYTHON=OFF \
  -DSHERPA_NCNN_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_NCNN_ENABLE_JNI=OFF \
  -DSHERPA_NCNN_ENABLE_BINARY=OFF \
  -DSHERPA_NCNN_ENABLE_TEST=OFF \
  -DSHERPA_NCNN_ENABLE_C_API=ON \
  -B build/simulator_x86_64

cmake --build build/os64 -j 4
# Generate headers for sherpa-ncnn.xcframework
cmake --build build/os64 --target install
# Clean files
rm -rf install/lib/cmake
rm -rf install/lib/pkgconfig
rm -rf install/include/ncnn
rm -rf install/include/kaldi-native-fbank
cmake --build build/simulator_arm64 -j 8
cmake --build build/simulator_x86_64 -j 8

# For sherpa-ncnn.xcframework
rm -rf sherpa-ncnn.xcframework

libtool -static -o build/os64/sherpa-ncnn.a \
  build/os64/lib/libncnn.a \
  build/os64/lib/libsherpa-ncnn-c-api.a \
  build/os64/lib/libsherpa-ncnn-core.a \
  build/os64/lib/libkaldi-native-fbank-core.a

mkdir -p "build/simulator/lib"

for f in libncnn.a libsherpa-ncnn-c-api.a libsherpa-ncnn-core.a libkaldi-native-fbank-core.a; do
  lipo -create build/simulator_arm64/lib/${f} \
               build/simulator_x86_64/lib/${f} \
       -output build/simulator/lib/${f}
done

# Merge archive first, because the following xcodebuild create xcframework 
# cann't accept multi archive with the same architecture.
libtool -static -o build/simulator/sherpa-ncnn.a \
  build/simulator/lib/libncnn.a \
  build/simulator/lib/libsherpa-ncnn-c-api.a \
  build/simulator/lib/libsherpa-ncnn-core.a \
  build/simulator/lib/libkaldi-native-fbank-core.a


xcodebuild -create-xcframework \
      -library "build/os64/sherpa-ncnn.a" \
      -library "build/simulator/sherpa-ncnn.a" \
      -output sherpa-ncnn.xcframework

# Copy Headers
mkdir -p sherpa-ncnn.xcframework/Headers
cp -av install/include/* sherpa-ncnn.xcframework/Headers
