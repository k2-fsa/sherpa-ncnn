function(download_kaldi_native_fbank)
  include(FetchContent)

  set(kaldi_native_fbank_URL  "https://github.com/csukuangfj/kaldi-native-fbank/archive/refs/tags/v1.8.tar.gz")
  set(kaldi_native_fbank_HASH "SHA256=1aefda7e76456e2c5005e05fa10cdc7f91c3ffc0a7fad000a90b50e7ed18087b")

  # If you don't have access to the internet, please download it to your
  # local drive and modify the following line according to your needs.
  # set(kaldi_native_fbank_URL  "file:///star-fj/fangjun/download/github/kaldi-native-fbank-1.8.tar.gz")

  set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(kaldi_native_fbank
    URL               ${kaldi_native_fbank_URL}
    URL_HASH          ${kaldi_native_fbank_HASH}
  )

  FetchContent_GetProperties(kaldi_native_fbank)
  if(NOT kaldi_native_fbank_POPULATED)
    message(STATUS "Downloading kaldi-native-fbank ${kaldi_native_fbank_URL}")
    FetchContent_Populate(kaldi_native_fbank)
  endif()
  message(STATUS "kaldi-native-fbank is downloaded to ${kaldi_native_fbank_SOURCE_DIR}")
  message(STATUS "kaldi-native-fbank's binary dir is ${kaldi_native_fbank_BINARY_DIR}")

  add_subdirectory(${kaldi_native_fbank_SOURCE_DIR} ${kaldi_native_fbank_BINARY_DIR})
  install(TARGETS kaldi-native-fbank-core DESTINATION lib)

  target_include_directories(kaldi-native-fbank-core
    INTERFACE
      ${kaldi_native_fbank_SOURCE_DIR}/
  )
endfunction()

download_kaldi_native_fbank()

