cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(sherpa-ncnn)

set(SHERPA_NCNN_VERSION "2.1.7")

# Disable warning about
#
# "The DOWNLOAD_EXTRACT_TIMESTAMP option was not given and policy CMP0135 is
#  not set.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT APPLE)
  set(SHERPA_NCNN_RPATH_ORIGIN "$ORIGIN")
else()
  set(CMAKE_MACOSX_RPATH ON)
  set(SHERPA_NCNN_RPATH_ORIGIN "@loader_path")
endif()

set(CMAKE_INSTALL_RPATH ${SHERPA_NCNN_RPATH_ORIGIN})
set(CMAKE_BUILD_RPATH ${SHERPA_NCNN_RPATH_ORIGIN})

# You will find a file compile_commands.json in the build directory
# if it is enabled
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "Whether to build shared libraries" OFF)
option(SHERPA_NCNN_ENABLE_PYTHON "Whether to build Python" OFF)
option(SHERPA_NCNN_ENABLE_PORTAUDIO "Whether to build with portaudio" ON)
option(SHERPA_NCNN_ENABLE_JNI "Whether to build JNI internface" OFF)
option(SHERPA_NCNN_ENABLE_BINARY "Whether to build the binary sherpa-ncnn" ON)
option(SHERPA_NCNN_ENABLE_TEST "Whether to build tests" OFF)
option(SHERPA_NCNN_ENABLE_C_API "Whether to build C API" ON)
option(SHERPA_NCNN_ENABLE_WASM "Whether to enable WASM" OFF)
option(SHERPA_NCNN_ENABLE_GENERATE_INT8_SCALE_TABLE "Whether to generate-int8-scale-table" ON)
option(SHERPA_NCNN_ENABLE_FFMPEG_EXAMPLES "Whether to enable ffmpeg-examples" OFF)

if(DEFINED ANDROID_ABI)
  message(STATUS "Set SHERPA_NCNN_ENABLE_JNI to ON for Android")
  set(SHERPA_NCNN_ENABLE_JNI ON CACHE BOOL "" FORCE)
endif()

# See
# https://stackoverflow.com/questions/33062728/cmake-link-shared-library-on-windows
if(BUILD_SHARED_LIBS AND MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(NOT BUILD_SHARED_LIBS AND MSVC)
  # see https://cmake.org/cmake/help/latest/prop_tgt/MSVC_RUNTIME_LIBRARY.html
  # https://stackoverflow.com/questions/14172856/compile-with-mt-instead-of-md-using-cmake
  if(MSVC)
      add_compile_options(
          $<$<CONFIG:>:/MT> #---------|
          $<$<CONFIG:Debug>:/MTd> #---|-- Statically link the runtime libraries
          $<$<CONFIG:Release>:/MT> #--|
      )
  endif()
endif()

message(STATUS "CMAKE_EXPORT_COMPILE_COMMANDS: ${CMAKE_EXPORT_COMPILE_COMMANDS}")
message(STATUS "BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}")
message(STATUS "SHERPA_NCNN_ENABLE_PYTHON ${SHERPA_NCNN_ENABLE_PYTHON}")
message(STATUS "SHERPA_NCNN_ENABLE_PORTAUDIO ${SHERPA_NCNN_ENABLE_PORTAUDIO}")
message(STATUS "SHERPA_NCNN_ENABLE_JNI ${SHERPA_NCNN_ENABLE_JNI}")
message(STATUS "SHERPA_NCNN_ENABLE_BINARY ${SHERPA_NCNN_ENABLE_BINARY}")
message(STATUS "SHERPA_NCNN_ENABLE_TEST ${SHERPA_NCNN_ENABLE_TEST}")
message(STATUS "SHERPA_NCNN_ENABLE_C_API ${SHERPA_NCNN_ENABLE_C_API}")
message(STATUS "SHERPA_NCNN_ENABLE_GENERATE_INT8_SCALE_TABLE ${SHERPA_NCNN_ENABLE_GENERATE_INT8_SCALE_TABLE}")
message(STATUS "SHERPA_NCNN_ENABLE_FFMPEG_EXAMPLES ${SHERPA_NCNN_ENABLE_FFMPEG_EXAMPLES}")

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# Set the release compiler flags to support debugging, see https://github.com/k2-fsa/sherpa-ncnn/issues/147
if(SHERPA_NCNN_ENABLE_DEBUG_FOR_RELEASE)
  message(STATUS "Enable debugging for Release")
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " -g -O0")
endif()

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_CXX_EXTENSIONS OFF)

if(SHERPA_NCNN_ENABLE_BINARY AND UNIX AND NOT APPLE)
  include(CheckIncludeFileCXX)
  check_include_file_cxx(alsa/asoundlib.h SHERPA_NCNN_HAS_ALSA)
  if(SHERPA_NCNN_HAS_ALSA)
    add_definitions(-DSHERPA_NCNN_ENABLE_ALSA=1)
  elseif(UNIX AND NOT APPLE)
    message(WARNING "\
  Could not find alsa/asoundlib.h !
  We won't build sherpa-ncnn-alsa
  To fix that, please do:
    (1) sudo apt-get install alsa-utils libasound2-dev
    (2) rm -rf build
    (3) re-try
  ")
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

if(WIN32)
  add_definitions(-DNOMINMAX) # Otherwise, std::max() and std::min() won't work
endif()

if(WIN32 AND MSVC)
  # disable various warnings for MSVC
  # 4244: 'return': conversion from 'unsigned __int64' to 'int', possible loss of data
  # 4267: 'initializing': conversion from 'size_t' to 'int', possible loss of data
  # 4305: 'argument': truncation from 'double' to 'const float'
  # 4334: '<<': result of 32-bit shift implicitly converted to 64 bits
  # 4800: 'int': forcing value to bool 'true' or 'false'
  # 4996: 'fopen': This function or variable may be unsafe
  set(disabled_warnings
      /wd4244
      /wd4267
      /wd4305
      /wd4334
      /wd4800
      /wd4996
  )
  message(STATUS "Disabled warnings: ${disabled_warnings}")
  foreach(w IN LISTS disabled_warnings)
    string(APPEND CMAKE_CXX_FLAGS " ${w} ")
  endforeach()
endif()

include(kaldi-native-fbank)
include(ncnn)

if(SHERPA_NCNN_ENABLE_PORTAUDIO)
  include(portaudio)
endif()

if(SHERPA_NCNN_ENABLE_PYTHON)
  include(pybind11)
endif()

add_subdirectory(sherpa-ncnn)

if(SHERPA_NCNN_ENABLE_FFMPEG_EXAMPLES)
  add_subdirectory(ffmpeg-examples)
endif()

if(SHERPA_NCNN_ENABLE_C_API AND SHERPA_NCNN_ENABLE_BINARY)
  add_subdirectory(c-api-examples)
endif()

if(SHERPA_NCNN_ENABLE_WASM)
  add_subdirectory(wasm)
endif()

set(SHERPA_NCNN_PKG_CONFIG_EXTRA_LIBS)

if(NOT BUILD_SHARED_LIBS)
  if(APPLE)
    set(SHERPA_NCNN_PKG_CONFIG_EXTRA_LIBS "-lomp -lc++")
  endif()

  if(UNIX AND NOT APPLE)
    set(SHERPA_NCNN_PKG_CONFIG_EXTRA_LIBS "-lm -lstdc++ -fopenmp")
  endif()
endif()

# See https://people.freedesktop.org/~dbn/pkg-config-guide.html
configure_file(cmake/sherpa-ncnn.pc.in ${PROJECT_BINARY_DIR}/sherpa-ncnn.pc @ONLY)
install(
  FILES
    ${PROJECT_BINARY_DIR}/sherpa-ncnn.pc
  DESTINATION
    .
)
