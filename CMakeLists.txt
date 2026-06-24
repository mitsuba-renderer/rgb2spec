cmake_minimum_required (VERSION 3.10)
project(rgb2spec)

if (POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
endif()

# Set a default build configuration (Release)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(rgb2spec_opt rgb2spec_opt.cpp)
add_library(rgb2spec STATIC rgb2spec.c)

target_link_libraries(rgb2spec_opt PRIVATE Threads::Threads)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/srgb.coeff
  DEPENDS rgb2spec_opt
  COMMAND $<TARGET_FILE:rgb2spec_opt> 64 ${CMAKE_CURRENT_BINARY_DIR}/srgb.coeff
)

add_custom_target(
  rgb2spec_opt_run
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/srgb.coeff
)
