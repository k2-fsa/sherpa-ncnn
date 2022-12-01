function(download_portaudio)
  include(FetchContent)

  set(portaudio_URL  "http://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz")
  set(portaudio_HASH "SHA256=47efbf42c77c19a05d22e627d42873e991ec0c1357219c0d74ce6a2948cb2def")
  if(BUILD_SHARED_LIBS)
    set(PA_BUILD_SHARED ON CACHE BOOL "" FORCE)
    set(PA_BUILD_STATIC OFF CACHE BOOL "" FORCE)
  else()
    set(PA_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(PA_BUILD_STATIC ON CACHE BOOL "" FORCE)
  endif()

  FetchContent_Declare(portaudio
    URL               ${portaudio_URL}
    URL_HASH          ${portaudio_HASH}
  )

  FetchContent_GetProperties(portaudio)
  if(NOT portaudio_POPULATED)
    message(STATUS "Downloading portaudio")
    FetchContent_Populate(portaudio)
  endif()
  message(STATUS "portaudio is downloaded to ${portaudio_SOURCE_DIR}")
  message(STATUS "portaudio's binary dir is ${portaudio_BINARY_DIR}")

  if(APPLE)
    set(CMAKE_MACOSX_RPATH ON) # to solve the following warning on macOS
  endif()

  add_subdirectory(${portaudio_SOURCE_DIR} ${portaudio_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction()

download_portaudio()

# Note
# See http://portaudio.com/docs/v19-doxydocs/tutorial_start.html
# for how to use portaudio
