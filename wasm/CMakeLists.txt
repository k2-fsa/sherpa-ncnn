if(NOT $ENV{SHERPA_NCNN_IS_USING_BUILD_WASM_SH})
  message(FATAL_ERROR "Please use ./build-wasm.sh to build for wasm")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets/decoder_jit_trace-pnnx.ncnn.bin")
  message(WARNING "${CMAKE_CURRENT_SOURCE_DIR}/assets/decoder_jit_trace-pnnx.ncnn.bin does not exist")
  message(FATAL_ERROR "Please read ${CMAKE_CURRENT_SOURCE_DIR}/assets/README.md before you continue")
endif()

set(exported_functions
  AcceptWaveform
  CreateRecognizer
  CreateStream
  Decode
  DestroyRecognizer
  DestroyResult
  DestroyStream
  GetResult
  InputFinished
  IsEndpoint
  IsReady
  Reset
  )
set(mangled_exported_functions)
foreach(x IN LISTS exported_functions)
  list(APPEND mangled_exported_functions "_${x}")
endforeach()

list(JOIN mangled_exported_functions "," all_exported_functions)

include_directories(${CMAKE_SOURCE_DIR})
set(MY_FLAGS " -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=512MB ")
string(APPEND MY_FLAGS " -sEXPORTED_FUNCTIONS=[_CopyHeap,_malloc,_free,${all_exported_functions}] ")
string(APPEND MY_FLAGS "--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/assets@. ")
string(APPEND MY_FLAGS " -sEXPORTED_RUNTIME_METHODS=['ccall','stringToUTF8','setValue','getValue'] ")
message(STATUS "MY_FLAGS: ${MY_FLAGS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MY_FLAGS}")
set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} ${MY_FLAGS}")

add_executable(sherpa-ncnn-wasm-main sherpa-ncnn-wasm-main.cc)
target_link_libraries(sherpa-ncnn-wasm-main sherpa-ncnn-core sherpa-ncnn-c-api)
install(TARGETS sherpa-ncnn-wasm-main DESTINATION bin/wasm)

install(
  FILES
    "sherpa-ncnn.js"
    "app.js"
    "index.html"
    "$<TARGET_FILE_DIR:sherpa-ncnn-wasm-main>/sherpa-ncnn-wasm-main.js"
    "$<TARGET_FILE_DIR:sherpa-ncnn-wasm-main>/sherpa-ncnn-wasm-main.wasm"
    "$<TARGET_FILE_DIR:sherpa-ncnn-wasm-main>/sherpa-ncnn-wasm-main.data"
  DESTINATION
    bin/wasm
)
