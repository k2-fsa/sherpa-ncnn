add_executable(online-fbank-test online-fbank-test.cc)
target_link_libraries(online-fbank-test kaldi-native-fbank-core)

add_executable(sherpa-ncnn
  sherpa-ncnn.cc
)
target_link_libraries(sherpa-ncnn ncnn)
