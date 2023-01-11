cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(sherpa)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

add_subdirectory(kaldi-native-fbank)
add_subdirectory(sherpa-ncnn)
