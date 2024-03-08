
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
)

if(SHERPA_NCNN_HAS_ALSA)
  list(APPEND srcs ${CMAKE_SOURCE_DIR}/sherpa-ncnn/csrc/alsa.cc alsa.cc)
else()
  list(APPEND srcs faked-alsa.cc)
endif()

pybind11_add_module(_sherpa_ncnn ${srcs})
target_link_libraries(_sherpa_ncnn PRIVATE sherpa-ncnn-core)

if(APPLE)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_DIR
  )
  message(STATUS "PYTHON_SITE_PACKAGE_DIR: ${PYTHON_SITE_PACKAGE_DIR}")
  target_link_libraries(_sherpa_ncnn PRIVATE "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}")
endif()

if(NOT WIN32)
  target_link_libraries(_sherpa_ncnn PRIVATE "-Wl,-rpath,${SHERPA_NCNN_RPATH_ORIGIN}/sherpa_ncnn/lib")
endif()

if(SHERPA_NCNN_HAS_ALSA)
  if(DEFINED ENV{SHERPA_NCNN_ALSA_LIB_DIR})
    target_link_libraries(_sherpa_ncnn PRIVATE -L$ENV{SHERPA_NCNN_ALSA_LIB_DIR} -lasound)
  else()
    target_link_libraries(_sherpa_ncnn PRIVATE asound)
  endif()
endif()

install(TARGETS _sherpa_ncnn
  DESTINATION ../
)
