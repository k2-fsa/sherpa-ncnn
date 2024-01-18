function(download_kaldi_native_fbank)
  include(FetchContent)

  # Please also change ../pack-for-embedded-systems.sh
  set(kaldi_native_fbank_URL  "https://github.com/csukuangfj/kaldi-native-fbank/archive/refs/tags/v1.18.6.tar.gz")
  set(kaldi_native_fbank_URL2  "https://hub.nuaa.cf/csukuangfj/kaldi-native-fbank/archive/refs/tags/v1.18.6.tar.gz")
  set(kaldi_native_fbank_HASH "SHA256=6202a00cd06ba8ff89beb7b6f85cda34e073e94f25fc29e37c519bff0706bf19")

  # If you don't have access to the Internet, please download it to your
  # local drive and modify the following line according to your needs.
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldi-native-fbank-1.18.6.tar.gz
    $ENV{HOME}/asr/kaldi-native-fbank-1.18.6.tar.gz
    ${PROJECT_SOURCE_DIR}/kaldi-native-fbank-1.18.6.tar.gz
    ${PROJECT_BINARY_DIR}/kaldi-native-fbank-1.18.6.tar.gz
    /tmp/kaldi-native-fbank-1.18.6.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldi_native_fbank_URL  "${f}")
      file(TO_CMAKE_PATH "${kaldi_native_fbank_URL}" kaldi_native_fbank_URL)
      set(kaldi_native_fbank_URL2)
      break()
    endif()
  endforeach()

  set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_FBANK_ENABLE_CHECK OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(kaldi_native_fbank
    URL
      ${kaldi_native_fbank_URL}
      ${kaldi_native_fbank_URL2}
    URL_HASH          ${kaldi_native_fbank_HASH}
  )

  FetchContent_GetProperties(kaldi_native_fbank)
  if(NOT kaldi_native_fbank_POPULATED)
    message(STATUS "Downloading kaldi-native-fbank from ${kaldi_native_fbank_URL}")
    FetchContent_Populate(kaldi_native_fbank)
  endif()
  message(STATUS "kaldi-native-fbank is downloaded to ${kaldi_native_fbank_SOURCE_DIR}")
  message(STATUS "kaldi-native-fbank's binary dir is ${kaldi_native_fbank_BINARY_DIR}")

  add_subdirectory(${kaldi_native_fbank_SOURCE_DIR} ${kaldi_native_fbank_BINARY_DIR} EXCLUDE_FROM_ALL)
  if(SHERPA_NCNN_ENABLE_PYTHON AND WIN32)
    install(TARGETS kaldi-native-fbank-core DESTINATION ..)
  else()
    install(TARGETS kaldi-native-fbank-core DESTINATION lib)
  endif()

  target_include_directories(kaldi-native-fbank-core
    INTERFACE
      ${kaldi_native_fbank_SOURCE_DIR}/
  )
endfunction()

download_kaldi_native_fbank()

