add_subdirectory(csrc)

if(DEFINED ANDROID_ABI)
  add_subdirectory(jni)
endif()

if(SHERPA_NCNN_ENABLE_PYTHON)
  add_subdirectory(python)
endif()
