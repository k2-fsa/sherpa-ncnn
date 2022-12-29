
include_directories(${PROJECT_SOURCE_DIR})
set(srcs
  decode.cc
  features.cc
  mat-util.cc
  model.cc
  sherpa-ncnn.cc
)

pybind11_add_module(_sherpa_ncnn ${srcs})
target_link_libraries(_sherpa_ncnn PRIVATE sherpa-ncnn-core)
target_link_libraries(_sherpa_ncnn PRIVATE ncnn)
