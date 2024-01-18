function(download_ncnn)
  include(FetchContent)

  # We use a modified version of NCNN.
  # The changed code is in
  # https://github.com/csukuangfj/ncnn/pull/7

  # Please also change ../pack-for-embedded-systems.sh
  set(ncnn_URL  "https://github.com/csukuangfj/ncnn/archive/refs/tags/sherpa-1.1.tar.gz")
  set(ncnn_URL2 "https://hub.nuaa.cf/csukuangfj/ncnn/archive/refs/tags/sherpa-1.1.tar.gz")
  set(ncnn_HASH "SHA256=254aaedf8ad3e6baaa63bcd5d23e9673e3973d7cb2154c18e5c7743d45b4e160")

  # If you don't have access to the Internet, please download it to your
  # local drive and modify the following line according to your needs.
  set(possible_file_locations
    $ENV{HOME}/Downloads/ncnn-sherpa-1.1.tar.gz
    $ENV{HOME}/asr/ncnn-sherpa-1.1.tar.gz
    ${PROJECT_SOURCE_DIR}/ncnn-sherpa-1.1.tar.gz
    ${PROJECT_BINARY_DIR}/ncnn-sherpa-1.1.tar.gz
    /tmp/ncnn-sherpa-1.1.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(ncnn_URL  "${f}")
      file(TO_CMAKE_PATH "${ncnn_URL}" ncnn_URL)
      set(ncnn_URL2)
      break()
    endif()
  endforeach()

  if(NOT WIN32)
    FetchContent_Declare(ncnn
      URL
        ${ncnn_URL}
        ${ncnn_URL2}
      URL_HASH          ${ncnn_HASH}
      PATCH_COMMAND
        sed -i.bak "/ncnn PROPERTIES VERSION/d" "src/CMakeLists.txt"
    )
  else()
    FetchContent_Declare(ncnn
      URL
        ${ncnn_URL}
        ${ncnn_URL2}
      URL_HASH          ${ncnn_HASH}
    )
  endif()

  set(NCNN_PIXEL OFF CACHE BOOL "" FORCE)
  set(NCNN_PIXEL_ROTATE OFF CACHE BOOL "" FORCE)
  set(NCNN_PIXEL_AFFINE OFF CACHE BOOL "" FORCE)
  set(NCNN_PIXEL_DRAWING OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)

  set(NCNN_SHARED_LIB ${BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)

  set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)

  # For RNN-T with ScaledLSTM, the following operators are not sued,
  # so we keep them from compiling.
  #
  # CAUTION: If you switch to a different model, please change
  # the following disabled layers accordingly; otherwise, you
  # will get segmentation fault during runtime.
  set(disabled_layers
    AbsVal
    ArgMax
    BatchNorm
    Bias
    BNLL
    # Concat
    # Convolution
    # Crop
    Deconvolution
    # Dropout
    Eltwise
    ELU
    # Embed
    Exp
    # Flatten  # needed by innerproduct
    # InnerProduct
    # Input
    Log
    LRN
    # MemoryData
    MVN
    Pooling
    Power
    PReLU
    Proposal
    # Reduction
    # ReLU
    # Reshape
    ROIPooling
    Scale
    # Sigmoid
    # Slice
    # Softmax
    # Split
    SPP
    # TanH
    Threshold
    Tile
    # RNN
    # LSTM
    # BinaryOp
    # UnaryOp
    ConvolutionDepthWise
    # Padding # required by innerproduct and convolution
    Squeeze
    # ExpandDims
    Normalize
    # Permute
    PriorBox
    DetectionOutput
    Interp
    DeconvolutionDepthWise
    ShuffleChannel
    InstanceNorm
    Clip
    Reorg
    YoloDetectionOutput
    # Quantize
    # Dequantize
    Yolov3DetectionOutput
    PSROIPooling
    ROIAlign
    # Packing
    # Requantize
    # Cast  # needed InnerProduct
    HardSigmoid
    SELU
    HardSwish
    Noop
    PixelShuffle
    DeepCopy
    Mish
    StatisticsPooling
    Swish
    # Gemm
    GroupNorm
    LayerNorm
    Softplus
    GRU
    MultiHeadAttention
    GELU
    # Convolution1D
    Pooling1D
    # ConvolutionDepthWise1D
    Convolution3D
    ConvolutionDepthWise3D
    Pooling3D
    # MatMul
    Deconvolution1D
    # DeconvolutionDepthWise1D
    Deconvolution3D
    DeconvolutionDepthWise3D
    Einsum
    DeformableConv2D
    RelPositionalEncoding
    MakePadMask
    RelShift
    # GLU
    Fold
    Unfold
    GridSample
    CumulativeSum
    CopyTo
  )

  foreach(layer IN LISTS disabled_layers)
    string(TOLOWER ${layer} name)
    set(WITH_LAYER_${name} OFF CACHE BOOL "" FORCE)
  endforeach()

  FetchContent_GetProperties(ncnn)
  if(NOT ncnn_POPULATED)
    message(STATUS "Downloading ncnn from ${ncnn_URL}")
    FetchContent_Populate(ncnn)
  endif()
  message(STATUS "ncnn is downloaded to ${ncnn_SOURCE_DIR}")
  message(STATUS "ncnn's binary dir is ${ncnn_BINARY_DIR}")

  add_subdirectory(${ncnn_SOURCE_DIR} ${ncnn_BINARY_DIR} EXCLUDE_FROM_ALL)
  if(SHERPA_NCNN_ENABLE_PYTHON AND WIN32)
    install(TARGETS ncnn DESTINATION ..)
  else()
    install(TARGETS ncnn DESTINATION lib)
  endif()
endfunction()

download_ncnn()
