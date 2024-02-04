if(NOT $ENV{SHERPA_NCNN_IS_USING_BUILD_WASM_SH})
  message(FATAL_ERROR "Please use ./build-wasm.sh to build for wasm")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets/encoder_jit_trace-pnnx.ncnn.bin")
  message(FATAL_ERROR "Please read ${CMAKE_CURRENT_SOURCE_DIR}/assets/README.md before you continue")
endif()


include_directories(${CMAKE_SOURCE_DIR})
set(MY_FLAGS " -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=512MB -s EXIT_RUNTIME=1 ")
string(APPEND MY_FLAGS " -sEXPORTED_FUNCTIONS=['_CopyHeap','_RunSherpaNcnn','_malloc','_free','_MyTest','_ReadData','_MyTest2'] ")
string(APPEND MY_FLAGS "--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/assets@. ")
string(APPEND MY_FLAGS " -sEXPORTED_RUNTIME_METHODS=['ccall','stringToUTF8','setValue'] ")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MY_FLAGS}")
set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} ${MY_FLAGS}")

if (NOT CMAKE_EXECUTABLE_SUFFIX STREQUAL ".js")
  message(FATAL_ERROR "The default suffix for building executables should be .js!")
endif()
# set(CMAKE_EXECUTABLE_SUFFIX ".html")

add_executable(sherpa-ncnn-wasm-main sherpa-ncnn-wasm-main.cc)
target_link_libraries(sherpa-ncnn-wasm-main sherpa-ncnn-core sherpa-ncnn-c-api)
install(TARGETS sherpa-ncnn-wasm-main DESTINATION bin/wasm)

install(
  FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/index.html
  DESTINATION
    bin/wasm
)

install(
  FILES
    "$<TARGET_FILE_DIR:sherpa-ncnn-wasm-main>/sherpa-ncnn-wasm-main.js"
    "sherpa-ncnn.js"
    "$<TARGET_FILE_DIR:sherpa-ncnn-wasm-main>/sherpa-ncnn-wasm-main.wasm"
    "$<TARGET_FILE_DIR:sherpa-ncnn-wasm-main>/sherpa-ncnn-wasm-main.data"
    # "$<TARGET_FILE_DIR:sherpa-ncnn-wasm-main>/sherpa-ncnn-wasm-main.html"
  DESTINATION
    bin/wasm
)
