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
target_link_libraries(sherpa-ncnn-core PUBLIC kaldi-native-fbank-core ncnn)
install(TARGETS sherpa-ncnn-core DESTINATION lib)

add_executable(sherpa-ncnn sherpa-ncnn.cc)
target_link_libraries(sherpa-ncnn PRIVATE sherpa-ncnn-core)
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

    target_link_libraries(sherpa-ncnn-microphone PRIVATE ${PA_LIB} sherpa-ncnn-core)

    install(TARGETS sherpa-ncnn-microphone DESTINATION bin)

    install(TARGETS ${PA_LIB} DESTINATION lib)
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

set(bins sherpa-ncnn)

if(NOT DEFINED ANDROID_ABI AND SHERPA_NCNN_ENABLE_PORTAUDIO)
  list(APPEND bins sherpa-ncnn-microphone)
endif()

if(NOT WIN32)
  foreach(bin IN LISTS bins)
    target_link_libraries(${bin} PRIVATE "-Wl,-rpath,${SHERPA_NCNN_RPATH_ORIGIN}/../lib")
    target_link_libraries(${bin} PRIVATE "-Wl,-rpath,${SHERPA_NCNN_RPATH_ORIGIN}/../lib/python3.6/site-packages/sherpa_ncnn/lib")
    target_link_libraries(${bin} PRIVATE "-Wl,-rpath,${SHERPA_NCNN_RPATH_ORIGIN}/../lib/python3.7/site-packages/sherpa_ncnn/lib")
    target_link_libraries(${bin} PRIVATE "-Wl,-rpath,${SHERPA_NCNN_RPATH_ORIGIN}/../lib/python3.8/site-packages/sherpa_ncnn/lib")
    target_link_libraries(${bin} PRIVATE "-Wl,-rpath,${SHERPA_NCNN_RPATH_ORIGIN}/../lib/python3.9/site-packages/sherpa_ncnn/lib")
    target_link_libraries(${bin} PRIVATE "-Wl,-rpath,${SHERPA_NCNN_RPATH_ORIGIN}/../lib/python3.10/site-packages/sherpa_ncnn/lib")
    target_link_libraries(${bin} PRIVATE "-Wl,-rpath,${SHERPA_NCNN_RPATH_ORIGIN}/../lib/python3.11/site-packages/sherpa_ncnn/lib")
    target_link_libraries(${bin} PRIVATE "-Wl,-rpath,${SHERPA_NCNN_RPATH_ORIGIN}/../lib/python3.12/site-packages/sherpa_ncnn/lib")
  endforeach()
endif()
