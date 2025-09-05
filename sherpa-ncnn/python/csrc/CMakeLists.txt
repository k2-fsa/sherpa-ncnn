
include_directories(${PROJECT_SOURCE_DIR})
set(srcs
  decoder.cc
  display.cc
  endpoint.cc
  features.cc
  model.cc
  recognizer.cc
  sherpa-ncnn.cc
  stream.cc
  offline-tts-vits-model-config.cc
)
list(APPEND srcs
  offline-tts-model-config.cc
  offline-tts-vits-model-config.cc
)

if(SHERPA_NCNN_HAS_ALSA)
  list(APPEND srcs ${CMAKE_SOURCE_DIR}/sherpa-ncnn/csrc/alsa.cc alsa.cc)
else()
  list(APPEND srcs faked-alsa.cc)
endif()

pybind11_add_module(_sherpa_ncnn ${srcs})
target_link_libraries(_sherpa_ncnn PRIVATE sherpa-ncnn-core)

if(SHERPA_NCNN_HAS_ALSA)
  if(DEFINED ENV{SHERPA_NCNN_ALSA_LIB_DIR})
    target_link_libraries(_sherpa_ncnn PRIVATE -L$ENV{SHERPA_NCNN_ALSA_LIB_DIR} -lasound)
  else()
    target_link_libraries(_sherpa_ncnn PRIVATE asound)
  endif()
endif()

install(TARGETS _sherpa_ncnn DESTINATION lib)
