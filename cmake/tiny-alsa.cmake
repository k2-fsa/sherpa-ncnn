function(download_tiny_alsa)
  include(FetchContent)

  set(tiny_alsa_URL  "https://github.com/tinyalsa//tinyalsa/archive/4fbaeef03cd1cb216e0f356c0433ca70f8b9c464.zip")
  set(tiny_alsa_URL2 "https://huggingface.co/csukuangfj/sherpa-ncnn-cmake-deps/resolve/main/tinyalsa-4fbaeef03cd1cb216e0f356c0433ca70f8b9c464.zip")
  set(tiny_alsa_HASH "SHA256=13a0e55fef7fa114db843b2ce51bf406f1b6861aed4340984adb13f49bd8457c")

  # If you don't have access to the Internet, please download it to your
  # local drive and modify the following line according to your needs.
  set(possible_file_locations
    $ENV{HOME}/Downloads/tinyalsa-4fbaeef03cd1cb216e0f356c0433ca70f8b9c464.zip
    $ENV{HOME}/asr/tinyalsa-4fbaeef03cd1cb216e0f356c0433ca70f8b9c464.zip
    ${PROJECT_SOURCE_DIR}/tinyalsa-4fbaeef03cd1cb216e0f356c0433ca70f8b9c464.zip
    ${PROJECT_BINARY_DIR}/tinyalsa-4fbaeef03cd1cb216e0f356c0433ca70f8b9c464.zip
    /tmp/tinyalsa-4fbaeef03cd1cb216e0f356c0433ca70f8b9c464.zip
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(tiny_alsa_URL  "${f}")
      file(TO_CMAKE_PATH "${tiny_alsa_URL}" tiny_alsa_URL)
      set(tiny_alsa_URL2)
      break()
    endif()
  endforeach()

  FetchContent_Declare(tiny_alsa
    URL
      ${tiny_alsa_URL}
      ${tiny_alsa_URL2}
    URL_HASH          ${tiny_alsa_HASH}
  )

  set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)
  set(TINYALSA_USES_PLUGINS OFF CACHE BOOL "" FORCE)
  set(TINYALSA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(TINYALSA_BUILD_UTILS OFF CACHE BOOL "" FORCE)

  FetchContent_GetProperties(tiny_alsa)
  if(NOT tiny_alsa_POPULATED)
    message(STATUS "Downloading tiny_alsa from ${tiny_alsa_URL}")
    FetchContent_Populate(tiny_alsa)
  endif()
  message(STATUS "tiny_alsa is downloaded to ${tiny_alsa_SOURCE_DIR}")
  message(STATUS "tiny_alsa's binary dir is ${tiny_alsa_BINARY_DIR}")

  add_subdirectory(${tiny_alsa_SOURCE_DIR} ${tiny_alsa_BINARY_DIR})
endfunction()

download_tiny_alsa()
