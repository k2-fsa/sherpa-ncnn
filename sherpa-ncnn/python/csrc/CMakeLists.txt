
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


install(TARGETS _sherpa_ncnn
  DESTINATION ../
)
