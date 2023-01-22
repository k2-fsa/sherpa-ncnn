include_directories(${CMAKE_SOURCE_DIR})
add_library(sherpa-ncnn-c-api c-api.cc)
target_link_libraries(sherpa-ncnn-c-api sherpa-ncnn-core)

install(TARGETS sherpa-ncnn-c-api DESTINATION lib)

install(FILES c-api.h
  DESTINATION include/sherpa-ncnn/
)
