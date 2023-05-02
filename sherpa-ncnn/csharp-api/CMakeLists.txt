find_program(DOTNET_EXECUTABLE NAMES dotnet)
if(NOT DOTNET_EXECUTABLE)
  message(FATAL_ERROR "Cannot find the executable dotnet.")
endif()

message(STATUS "DOTNET_EXECUTABLE: ${DOTNET_EXECUTABLE}")

# Runtime IDentifier
# see: https://docs.microsoft.com/en-us/dotnet/core/rid-catalog
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
  set(SHERPA_NCNN_DOTNET_PLATFORM arm64)
else()
  set(SHERPA_NCNN_DOTNET_PLATFORM x64)
endif()

if(APPLE)
  set(SHERPA_NCNN_DOTNET_RID osx-${SHERPA_NCNN_DOTNET_PLATFORM})
elseif(UNIX)
  set(SHERPA_NCNN_DOTNET_RID linux-${SHERPA_NCNN_DOTNET_PLATFORM})
elseif(WIN32)
  set(SHERPA_NCNN_DOTNET_RID win-${SHERPA_NCNN_DOTNET_PLATFORM})
else()
  message(FATAL_ERROR "Unsupported system ${CMAKE_SYSTEM_NAME}!")
endif()

message(STATUS ".Net RID: ${SHERPA_NCNN_DOTNET_RID}")
set(MY_CURRENT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}")

configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/sherpa-ncnn.csproj.runtime.in
  ${MY_CURRENT_BINARY_DIR}/${SHERPA_NCNN_DOTNET_RID}/sherpa-ncnn.csproj.in
  @ONLY)

file(GENERATE
  OUTPUT ${MY_CURRENT_BINARY_DIR}/${SHERPA_NCNN_DOTNET_RID}/sherpa-ncnn.csproj
  INPUT ${MY_CURRENT_BINARY_DIR}/${SHERPA_NCNN_DOTNET_RID}/sherpa-ncnn.csproj.in)

add_custom_command(
  OUTPUT ${MY_CURRENT_BINARY_DIR}/${SHERPA_NCNN_DOTNET_RID}/timestamp
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_LIST_DIR}/README.md ${MY_CURRENT_BINARY_DIR}/${SHERPA_NCNN_DOTNET_RID}/README.md
  COMMAND ${CMAKE_COMMAND} -E env --unset=TARGETNAME ${DOTNET_EXECUTABLE} build -c Release /p:Platform=${SHERPA_NCNN_DOTNET_PLATFORM} sherpa-ncnn.csproj
  COMMAND ${CMAKE_COMMAND} -E env --unset=TARGETNAME ${DOTNET_EXECUTABLE} pack -c Release sherpa-ncnn.csproj
  COMMAND ${CMAKE_COMMAND} -E touch ${MY_CURRENT_BINARY_DIR}/${SHERPA_NCNN_DOTNET_RID}/timestamp
  DEPENDS
    ${MY_CURRENT_BINARY_DIR}/${SHERPA_NCNN_DOTNET_RID}/sherpa-ncnn.csproj
    ${CMAKE_CURRENT_LIST_DIR}/README.md
    sherpa-ncnn-c-api
  BYPRODUCTS
    ${MY_CURRENT_BINARY_DIR}/${SHERPA_NCNN_DOTNET_RID}/bin
    ${MY_CURRENT_BINARY_DIR}/${SHERPA_NCNN_DOTNET_RID}/obj
  VERBATIM
  COMMENT "Generate .Net native package ${SHERPA_NCNN_DOTNET_RID} (${MY_CURRENT_BINARY_DIR}/${SHERPA_NCNN_DOTNET_RID}/timestamp)"
  WORKING_DIRECTORY ${MY_CURRENT_BINARY_DIR}/${SHERPA_NCNN_DOTNET_RID})

add_custom_target(_sherpa-ncnn-native-dot-net-package
  DEPENDS
    ${MY_CURRENT_BINARY_DIR}/${SHERPA_NCNN_DOTNET_RID}/timestamp
  )

set(SHERPA_NCNN_DOTNET_PACKAGES_DIR "${PROJECT_BINARY_DIR}/packages")
message(STATUS "SHERPA_NCNN_DOTNET_PACKAGES_DIR ${SHERPA_NCNN_DOTNET_PACKAGES_DIR}")
file(MAKE_DIRECTORY ${SHERPA_NCNN_DOTNET_PACKAGES_DIR})

add_custom_command(
  OUTPUT ${SHERPA_NCNN_DOTNET_PACKAGES_DIR}/org.k2fsa.sherpa.ncnn.runtime.${SHERPA_NCNN_DOTNET_RID}.${SHERPA_NCNN_VERSION}.nupkg
  COMMAND
    ${CMAKE_COMMAND} -E copy bin/$<CONFIG>/org.k2fsa.sherpa.ncnn.runtime.${SHERPA_NCNN_DOTNET_RID}.${SHERPA_NCNN_VERSION}.nupkg ${SHERPA_NCNN_DOTNET_PACKAGES_DIR}/org.k2fsa.sherpa.ncnn.runtime.${SHERPA_NCNN_DOTNET_RID}.${SHERPA_NCNN_VERSION}.nupkg
  DEPENDS
    ${MY_CURRENT_BINARY_DIR}/${SHERPA_NCNN_DOTNET_RID}/timestamp
  WORKING_DIRECTORY ${MY_CURRENT_BINARY_DIR}/${SHERPA_NCNN_DOTNET_RID})

add_custom_target(sherpa-ncnn-native-dot-net-package
  DEPENDS
    ${SHERPA_NCNN_DOTNET_PACKAGES_DIR}/org.k2fsa.sherpa.ncnn.runtime.${SHERPA_NCNN_DOTNET_RID}.${SHERPA_NCNN_VERSION}.nupkg)

# Now for the wrapper

file(MAKE_DIRECTORY ${MY_CURRENT_BINARY_DIR}/wrapper)

configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/sherpa-ncnn.csproj.in
  ${MY_CURRENT_BINARY_DIR}/wrapper/sherpa-ncnn.csproj
  @ONLY)

add_custom_command(
  OUTPUT ${MY_CURRENT_BINARY_DIR}/wrapper/timestamp
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_LIST_DIR}/README.md ${MY_CURRENT_BINARY_DIR}/wrapper/README.md
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_LIST_DIR}/sherpa-ncnn.cs ${MY_CURRENT_BINARY_DIR}/wrapper/sherpa-ncnn.cs
  COMMAND ${CMAKE_COMMAND} -E env --unset=TARGETNAME ${DOTNET_EXECUTABLE} build -c Release sherpa-ncnn.csproj
  COMMAND ${CMAKE_COMMAND} -E env --unset=TARGETNAME ${DOTNET_EXECUTABLE} pack -c Release sherpa-ncnn.csproj
  COMMAND ${CMAKE_COMMAND} -E touch ${MY_CURRENT_BINARY_DIR}/wrapper/timestamp
  DEPENDS
    ${CMAKE_CURRENT_LIST_DIR}/sherpa-ncnn.csproj.in
    ${CMAKE_CURRENT_LIST_DIR}/README.md
    ${CMAKE_CURRENT_LIST_DIR}/sherpa-ncnn.cs
    _sherpa-ncnn-native-dot-net-package
  BYPRODUCTS
    ${MY_CURRENT_BINARY_DIR}/wrapper/bin
    ${MY_CURRENT_BINARY_DIR}/wrapper/obj
  VERBATIM
  COMMENT "Generate .Net package (${MY_CURRENT_BINARY_DIR}/wrapper/timestamp)"
  WORKING_DIRECTORY ${MY_CURRENT_BINARY_DIR}/wrapper)

add_custom_target(_sherpa-ncnn-dot-net-package
  DEPENDS
    ${MY_CURRENT_BINARY_DIR}/wrapper/timestamp
  )

add_custom_command(
  OUTPUT ${SHERPA_NCNN_DOTNET_PACKAGES_DIR}/org.k2fsa.sherpa.ncnn.${SHERPA_NCNN_VERSION}.nupkg
  COMMAND
    ${CMAKE_COMMAND} -E copy bin/$<CONFIG>/org.k2fsa.sherpa.ncnn.${SHERPA_NCNN_VERSION}.nupkg ${SHERPA_NCNN_DOTNET_PACKAGES_DIR}/org.k2fsa.sherpa.ncnn.${SHERPA_NCNN_VERSION}.nupkg
  DEPENDS
    ${MY_CURRENT_BINARY_DIR}/wrapper/timestamp
    sherpa-ncnn-native-dot-net-package
  WORKING_DIRECTORY ${MY_CURRENT_BINARY_DIR}/wrapper)

add_custom_target(sherpa-ncnn-dot-net-package
  DEPENDS
    ${SHERPA_NCNN_DOTNET_PACKAGES_DIR}/org.k2fsa.sherpa.ncnn.${SHERPA_NCNN_VERSION}.nupkg)
