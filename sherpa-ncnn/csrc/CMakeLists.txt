include_directories(${CMAKE_SOURCE_DIR})

set(sherpa_ncnn_core_srcs
  context-graph.cc
  conv-emformer-model.cc
  decoder.cc
  endpoint.cc
  features.cc
  greedy-search-decoder.cc
  hypothesis.cc
  lstm-model.cc
  meta-data.cc
  model.cc
  modified-beam-search-decoder.cc
  poolingmodulenoproj.cc
  recognizer.cc
  resample.cc
  simpleupsample.cc
  stack.cc
  stream.cc
  symbol-table.cc
  tensorasstrided.cc
  wave-reader.cc
  zipformer-model.cc
)
add_library(sherpa-ncnn-core ${sherpa_ncnn_core_srcs})
target_link_libraries(sherpa-ncnn-core PUBLIC kaldi-native-fbank-core ncnn)

if(SHERPA_NCNN_ENABLE_PYTHON AND WIN32)
  install(TARGETS sherpa-ncnn-core DESTINATION ..)
else()
  install(TARGETS sherpa-ncnn-core DESTINATION lib)
endif()

if(NOT SHERPA_NCNN_ENABLE_PYTHON)
  if(SHERPA_NCNN_ENABLE_BINARY)
    add_executable(sherpa-ncnn sherpa-ncnn.cc)
    target_link_libraries(sherpa-ncnn PRIVATE sherpa-ncnn-core)
    install(TARGETS sherpa-ncnn DESTINATION bin)

    if(SHERPA_NCNN_HAS_ALSA)
      add_executable(sherpa-ncnn-alsa sherpa-ncnn-alsa.cc alsa.cc)
      target_link_libraries(sherpa-ncnn-alsa PRIVATE sherpa-ncnn-core)

      if(DEFINED ENV{SHERPA_NCNN_ALSA_LIB_DIR})
        target_link_libraries(sherpa-ncnn-alsa PRIVATE -L$ENV{SHERPA_NCNN_ALSA_LIB_DIR} -lasound)
      else()
        target_link_libraries(sherpa-ncnn-alsa PRIVATE asound)
      endif()
      install(TARGETS sherpa-ncnn-alsa DESTINATION bin)
    endif()

    if(SHERPA_NCNN_ENABLE_PORTAUDIO)
      add_executable(sherpa-ncnn-microphone
        sherpa-ncnn-microphone.cc
        microphone.cc
      )

      if(BUILD_SHARED_LIBS)
        set(PA_LIB portaudio)
      else()
        set(PA_LIB portaudio_static)
      endif()

      target_link_libraries(sherpa-ncnn-microphone PRIVATE ${PA_LIB} sherpa-ncnn-core)

      install(TARGETS sherpa-ncnn-microphone DESTINATION bin)
    endif()

    if(SHERPA_NCNN_ENABLE_GENERATE_INT8_SCALE_TABLE)
      add_executable(generate-int8-scale-table generate-int8-scale-table.cc)
      target_link_libraries(generate-int8-scale-table sherpa-ncnn-core)
    endif()
  endif()
endif()

if(SHERPA_NCNN_ENABLE_TEST)
  add_executable(test-resample test-resample.cc)
  target_link_libraries(test-resample sherpa-ncnn-core)
endif()
