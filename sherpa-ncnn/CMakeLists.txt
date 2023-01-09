add_subdirectory(csrc)

if(SHERPA_NCNN_ENABLE_JNI)
  add_subdirectory(jni)
endif()

if(SHERPA_NCNN_ENABLE_PYTHON)
  add_subdirectory(python)
endif()
