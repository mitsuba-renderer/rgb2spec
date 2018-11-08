cmake_minimum_required (VERSION 3.2.0)
project(rgb2spec)

if (POLICY CMP0062)
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

if (TBB_INCLUDE_DIRS)
  add_definitions(-DRGB2SPEC_USE_TBB=1)
endif()

add_executable(rgb2spec_opt rgb2spec_opt.cpp)
add_library(rgb2spec STATIC rgb2spec.c)

if (TBB_INCLUDE_DIRS)
  target_link_libraries(rgb2spec_opt PRIVATE tbb)
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/srgb.coeff
  DEPENDS rgb2spec_opt
  COMMAND $<TARGET_FILE:rgb2spec_opt> 32 ${CMAKE_CURRENT_BINARY_DIR}/srgb.coeff
)

add_custom_target(
  optimize
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/srgb.coeff
)
