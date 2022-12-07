include_directories(${CMAKE_SOURCE_DIR})

set(sherpa_ncnn_core_srcs
  conv-emformer-model.cc
  decode.cc
  features.cc
  lstm-model.cc
  meta-data.cc
  model.cc
  symbol-table.cc
  wave-reader.cc
)
add_library(sherpa-ncnn-core ${sherpa_ncnn_core_srcs})
target_link_libraries(sherpa-ncnn-core kaldi-native-fbank-core ncnn)

add_executable(sherpa-ncnn sherpa-ncnn.cc)
target_link_libraries(sherpa-ncnn sherpa-ncnn-core)

add_executable(sherpa-ncnn-microphone
  sherpa-ncnn-microphone.cc
  microphone.cc
)
if(BUILD_SHARED_LIBS)
  set(PA_LIB portaudio)
else()
  set(PA_LIB portaudio_static)
endif()

target_link_libraries(sherpa-ncnn-microphone ${PA_LIB} sherpa-ncnn-core)
