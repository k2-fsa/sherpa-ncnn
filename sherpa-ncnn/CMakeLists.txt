add_subdirectory(csrc)

if(DEFINED ANDROID_ABI)
  add_subdirectory(jni)
endif()
