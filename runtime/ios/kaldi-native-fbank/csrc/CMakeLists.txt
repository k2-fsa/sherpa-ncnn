include_directories(${CMAKE_SOURCE_DIR})

set(kaldi-native-fbank_srcs
  feature-fbank.cc
  feature-functions.cc
  feature-window.cc
  fftsg.c
  log.cc
  mel-computations.cc
  online-feature.cc
  rfft.cc
)
add_library(kaldi-native-fbank ${kaldi-native-fbank_srcs})
