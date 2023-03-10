
include_directories(${CMAKE_SOURCE_DIR})
add_executable(sherpa-ncnn-ffmpeg sherpa-ncnn-ffmpeg.cc)
target_link_libraries(sherpa-ncnn-ffmpeg pthread m)

# Link libraries from sherpa-ncnn.
target_link_libraries(sherpa-ncnn-ffmpeg sherpa-ncnn-c-api sherpa-ncnn-core kaldi-native-fbank-core ncnn)
if(BUILD_SHARED_LIBS)
    target_link_libraries(sherpa-ncnn-ffmpeg "portaudio")
else()
    target_link_libraries(sherpa-ncnn-ffmpeg "portaudio_static")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
include_directories(${AVCODEC_INCLUDE_DIRS})
target_link_directories(sherpa-ncnn-ffmpeg PRIVATE ${AVCODEC_LIBRARY_DIRS})
target_link_libraries(sherpa-ncnn-ffmpeg ${AVCODEC_LIBRARIES})
# All libraries of FFmpeg shares the same include and library directory.
# Note that ${AVCODEC_LIBRARIES} equals to avcodec, but we add it for consistence.
target_link_libraries(sherpa-ncnn-ffmpeg avcodec avformat avfilter avutil swresample swscale avdevice)

