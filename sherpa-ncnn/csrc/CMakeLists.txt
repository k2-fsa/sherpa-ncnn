include_directories(${CMAKE_SOURCE_DIR})

set(sherpa_ncnn_core_srcs
  context-graph.cc
  conv-emformer-model.cc
  decoder.cc
  endpoint.cc
  features.cc
  file-utils.cc
  greedy-search-decoder.cc
  hypothesis.cc
  lstm-model.cc
  math.cc
  meta-data.cc
  model.cc
  modified-beam-search-decoder.cc
  parse-options.cc
  poolingmodulenoproj.cc
  recognizer.cc
  resample.cc
  simpleupsample.cc
  stack.cc
  stream.cc
  symbol-table.cc
  tensorasstrided.cc
  text-utils.cc
  version.cc
  wave-reader.cc
  wave-writer.cc
  zipformer-model.cc
)

list(APPEND sherpa_ncnn_core_srcs
  offline-ctc-greedy-search-decoder.cc
  offline-model-config.cc
  offline-recognizer-impl.cc
  offline-recognizer.cc
  offline-sense-voice-model-config.cc
  offline-sense-voice-model.cc
  offline-stream.cc
)

list(APPEND sherpa_ncnn_core_srcs
  lexicon.cc
  offline-tts-impl.cc
  offline-tts-model-config.cc
  offline-tts-vits-model-config.cc
  offline-tts-vits-model-meta-data.cc
  offline-tts-vits-model.cc
  offline-tts.cc
)

list(APPEND sherpa_ncnn_core_srcs
  circular-buffer.cc
  silero-vad-model-config.cc
  silero-vad-model.cc
  voice-activity-detector.cc
)

add_library(sherpa-ncnn-core STATIC ${sherpa_ncnn_core_srcs})
target_link_libraries(sherpa-ncnn-core PUBLIC
  kaldi-native-fbank-core
  kaldifst_core
  ncnn
  fst
  fstfar
)

if(NOT BUILD_SHARED_LIBS)
  install(TARGETS sherpa-ncnn-core DESTINATION lib)
endif()

if(SHERPA_NCNN_ENABLE_BINARY)
  add_executable(sherpa-ncnn sherpa-ncnn.cc)
  add_executable(sherpa-ncnn-offline sherpa-ncnn-offline.cc)
  add_executable(sherpa-ncnn-offline-tts sherpa-ncnn-offline-tts.cc)
  add_executable(sherpa-ncnn-vad sherpa-ncnn-vad.cc)

  add_executable(sherpa-ncnn-version sherpa-ncnn-version.cc version.cc)
  install(TARGETS sherpa-ncnn-version DESTINATION bin)

  set(main_exes
    sherpa-ncnn
    sherpa-ncnn-offline
    sherpa-ncnn-offline-tts
    sherpa-ncnn-vad
  )

  if(SHERPA_NCNN_HAS_ALSA)
    add_executable(sherpa-ncnn-alsa sherpa-ncnn-alsa.cc alsa.cc)

    if(DEFINED ENV{SHERPA_NCNN_ALSA_LIB_DIR})
      target_link_libraries(sherpa-ncnn-alsa -L$ENV{SHERPA_NCNN_ALSA_LIB_DIR} -lasound)
    else()
      target_link_libraries(sherpa-ncnn-alsa asound)
    endif()

    list(APPEND main_exes
      sherpa-ncnn-alsa
    )
  endif()

  if(SHERPA_NCNN_ENABLE_PORTAUDIO)
    add_executable(sherpa-ncnn-microphone
      sherpa-ncnn-microphone.cc
      microphone.cc
    )
    list(APPEND main_exes
      sherpa-ncnn-microphone
    )

    set(PA_LIB portaudio_static)

    target_link_libraries(sherpa-ncnn-microphone ${PA_LIB})
  endif()

  foreach(exe IN LISTS main_exes)
    target_link_libraries(${exe} sherpa-ncnn-core)
    install(TARGETS ${exe} DESTINATION bin)
  endforeach()
  if(NOT WIN32)
    foreach(exe IN LISTS main_exes)
      target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_NCNN_RPATH_ORIGIN}/../lib")
      target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_NCNN_RPATH_ORIGIN}/../../../sherpa_ncnn/lib")

      if(SHERPA_NCNN_ENABLE_PYTHON)
        target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_NCNN_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_ncnn/lib")
      else()
        foreach(ver in ITEMS 3.8 3.9 3.10 3.11 3.12 3.13 3.14)
          target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_NCNN_RPATH_ORIGIN}/../lib/python${ver}/site-packages/sherpa_ncnn/lib")
        endforeach()
      endif()
    endforeach()
  endif()

  if(SHERPA_NCNN_ENABLE_GENERATE_INT8_SCALE_TABLE)
    add_executable(generate-int8-scale-table generate-int8-scale-table.cc)
    target_link_libraries(generate-int8-scale-table sherpa-ncnn-core)
  endif()
endif()

if(SHERPA_NCNN_ENABLE_TEST)
  add_executable(test-resample test-resample.cc)
  target_link_libraries(test-resample sherpa-ncnn-core)
  add_executable(test-context-graph test-context-graph.cc)
  target_link_libraries(test-context-graph sherpa-ncnn-core)
endif()
