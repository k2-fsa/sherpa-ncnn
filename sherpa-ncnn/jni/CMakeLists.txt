include_directories(${CMAKE_SOURCE_DIR})

if(NOT DEFINED ANDROID_ABI)
  if(NOT DEFINED ENV{JAVA_HOME})
    message(FATAL_ERROR "Please set the environment variable JAVA_HOME")
  endif()
  include_directories($ENV{JAVA_HOME}/include)
  include_directories($ENV{JAVA_HOME}/include/linux)
  include_directories($ENV{JAVA_HOME}/include/darwin)
endif()

add_library(sherpa-ncnn-jni SHARED
  common.cc
  jni.cc
  offline-recognizer.cc
  offline-stream.cc
  wave-reader.cc
)

target_compile_definitions(sherpa-ncnn-jni PRIVATE SHERPA_NCNN_BUILD_SHARED_LIBS=1)
target_compile_definitions(sherpa-ncnn-jni PRIVATE SHERPA_NCNN_BUILD_MAIN_LIB=1)

target_link_libraries(sherpa-ncnn-jni sherpa-ncnn-core)
install(TARGETS sherpa-ncnn-jni DESTINATION lib)
