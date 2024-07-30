cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set extension name here
set(TARGET_NAME airport)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

find_package(Arrow REQUIRED)
find_package(ArrowFlight REQUIRED)

project(${TARGET_NAME})
include_directories(src/include)

find_package(Protobuf CONFIG REQUIRED)


include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${PROTO_BINARY_DIR})

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")


message("PROTO_BINARY_DIR: ${PROTO_BINARY_DIR}")
message("PROTOBUF_INCLUDE_DIRS: $<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")
include_directories(SYSTEM "${PROTO_BINARY_DIR}")
#include_directories(SYSTEM "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")

add_library(${TARGET_NAME} OBJECT
  "${CMAKE_CURRENT_LIST_DIR}/substrait/proto/substrait/algebra.proto"
  "${CMAKE_CURRENT_LIST_DIR}/substrait/proto/substrait/capabilities.proto"
  "${CMAKE_CURRENT_LIST_DIR}/substrait/proto/substrait/extended_expression.proto"
  "${CMAKE_CURRENT_LIST_DIR}/substrait/proto/substrait/function.proto"
  "${CMAKE_CURRENT_LIST_DIR}/substrait/proto/substrait/parameterized_types.proto"
  "${CMAKE_CURRENT_LIST_DIR}/substrait/proto/substrait/plan.proto"
  "${CMAKE_CURRENT_LIST_DIR}/substrait/proto/substrait/type.proto"
  "${CMAKE_CURRENT_LIST_DIR}/substrait/proto/substrait/type_expressions.proto"
  "${CMAKE_CURRENT_LIST_DIR}/substrait/proto/substrait/extensions/extensions.proto"
)

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(PROTO_IMPORT_DIRS "${CMAKE_CURRENT_LIST_DIR}/substrait/proto/")

protobuf_generate(
    TARGET ${TARGET_NAME}
    IMPORT_DIRS "${PROTO_IMPORT_DIRS}"
    OUT_VAR PROTO_SOURCES
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")

message("PROTO_SOURCES: ${PROTO_SOURCES}")

set(EXTENSION_SOURCES
  src/airport_extension.cpp
  src/airport_flight_stream.cpp
  src/to_substrait.cpp
  src/custom_extensions.cpp
  src/custom_extensions_generated.cpp
  ${PROTO_SOURCES}
)


build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})

#target_link_libraries(${EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto)
#target_link_libraries(${LOADABLE_EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto)

target_link_libraries(${EXTENSION_NAME}
  Arrow::arrow_static
  ArrowFlight::arrow_flight_static
  gRPC::grpc++
  protobuf::libprotobuf
)

#target_link_libraries(${LOADABLE_TARGET_NAME} PRIVATE Arrow::arrow_shared ArrowFlight::arrow_flight_shared)

build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

target_link_libraries(${LOADABLE_EXTENSION_NAME}
  Arrow::arrow_static
  ArrowFlight::arrow_flight_static
  gRPC::grpc++
)


install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
