include_directories(${CMAKE_SOURCE_DIR})

add_library(sherpa-ncnn-jni jni.cc)
target_link_libraries(sherpa-ncnn-jni sherpa-ncnn-core)
install(TARGETS sherpa-ncnn-jni DESTINATION lib)
