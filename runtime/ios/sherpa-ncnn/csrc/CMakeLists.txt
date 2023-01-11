include_directories(${CMAKE_SOURCE_DIR})

set(VULKAN_SDK /Users/dan/VulkanSDK/1.3.231.1)
set(Vulkan_INCLUDE_DIR ${VULKAN_SDK}/MoltenVK/include)
set(Vulkan_LIBRARY ${VULKAN_SDK}/MoltenVK/dylib/iOS/libMoltenVK.dylib)

include_directories(${Vulkan_INCLUDE_DIR})

set(ncnn_DIR /Users/dan/Models/ASR/sherpa/ncnn-sherpa-0.5/build-ios/install/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/../../kaldi-native-fbank/csrc)
include_directories(${ncnn_DIR}/../../../include/ncnn)

set(sherpa_srcs
  decode.cc
  features.cc
  lstm-model.cc
  symbol-table.cc
)
add_library(sherpa ${sherpa_srcs})
