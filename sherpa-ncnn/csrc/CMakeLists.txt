include_directories(${CMAKE_SOURCE_DIR})

set(sherpa_ncnn_core_srcs
  decode.cc
  features.cc
  lstm-model.cc
  symbol-table.cc
  wave-reader.cc
)
add_library(sherpa-ncnn-core ${sherpa_ncnn_core_srcs})
target_link_libraries(sherpa-ncnn-core kaldi-native-fbank-core ncnn)

add_executable(sherpa-ncnn sherpa-ncnn.cc)
target_link_libraries(sherpa-ncnn sherpa-ncnn-core)
