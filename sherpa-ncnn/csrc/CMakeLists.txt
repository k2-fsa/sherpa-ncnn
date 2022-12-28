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
set_target_properties(sherpa-ncnn-core PROPERTIES VERSION ${SHERPA_NCNN_VERSION} SOVERSION ${SHERPA_NCNN_VERSION_MAJOR})
install(TARGETS sherpa-ncnn-core DESTINATION lib)

add_executable(sherpa-ncnn sherpa-ncnn.cc)
target_link_libraries(sherpa-ncnn sherpa-ncnn-core)
install(TARGETS sherpa-ncnn DESTINATION bin)

if(NOT DEFINED ANDROID_ABI)

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

    target_link_libraries(sherpa-ncnn-microphone ${PA_LIB} sherpa-ncnn-core)

    install(TARGETS sherpa-ncnn-microphone DESTINATION bin)
  endif()
endif()

set(hdrs
  decode.h
  features.h
  model.h
  symbol-table.h
  wave-reader.h
)

install(FILES ${hdrs}
  DESTINATION include/sherpa-ncnn/csrc
)

add_executable(generate-int8-scale-table generate-int8-scale-table.cc)
target_link_libraries(generate-int8-scale-table sherpa-ncnn-core)
