function(download_ncnn)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    # FetchContent is available since 3.11,
    # we've copied it to ${CMAKE_SOURCE_DIR}/cmake/Modules
    # so that it can be used in lower CMake versions.
    message(STATUS "Use FetchContent provided by sherpa-ncnn")
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  set(ncnn_URL  "https://github.com/csukuangfj/ncnn/archive/refs/tags/sherpa-0.1.tar.gz")
  set(ncnn_HASH "SHA256=bd9669798846a2727eaa05c4ce156d19e8c729a0f6ee9277d5c4ded33fd38dff")

  FetchContent_Declare(ncnn
    URL               ${ncnn_URL}
    URL_HASH          ${ncnn_HASH}
  )

  set(NCNN_INSTALL_SDK OFF CACHE BOOL "" FORCE)
  set(NCNN_PIXEL OFF CACHE BOOL "" FORCE)
  set(NCNN_PIXEL_ROTATE OFF CACHE BOOL "" FORCE)
  set(NCNN_PIXEL_AFFINE OFF CACHE BOOL "" FORCE)
  set(NCNN_PIXEL_DRAWING OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)

  set(NCNN_INT8 OFF CACHE BOOL "" FORCE) # TODO(fangjun): enable it
  set(NCNN_BF16 OFF CACHE BOOL "" FORCE) # TODO(fangjun): enable it

  set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)

  FetchContent_GetProperties(ncnn)
  if(NOT ncnn_POPULATED)
    message(STATUS "Downloading ncnn")
    FetchContent_Populate(ncnn)
  endif()
  message(STATUS "ncnn is downloaded to ${ncnn_SOURCE_DIR}")
  message(STATUS "ncnn's binary dir is ${ncnn_BINARY_DIR}")

  add_subdirectory(${ncnn_SOURCE_DIR} ${ncnn_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction()

download_ncnn()
