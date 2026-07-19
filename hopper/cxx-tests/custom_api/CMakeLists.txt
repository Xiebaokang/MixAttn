### We first make a complete CMakeLists.txt file here
### And then later consider how we could split it into smaller files

cmake_minimum_required(VERSION 3.26)


#################### PROLOGUE BEGIN ####################
# check if ccache exists
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  # use ccache
  set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
  set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
endif()

# check if mold exists
find_program(MOLD_FOUND mold)
if(MOLD_FOUND)
  # use mold linker
  set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=mold")
  set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=mold")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# fucking stupid pytorch relies on its custom VAR to specify CUDA architectures
# and it completely ignores CMAKE_CUDA_ARCHITECTURES

# the "TORCH_TORCH_CUDA_ARCH_LIST" is meant to choose CUDA arch like the below
# ``TORCH_CUDA_ARCH_LIST="6.1 8.6" python build_my_extension.py``
# ``TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" python build_my_extension.py``

# we implement our own mapping logic

# set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA architectures to build for" FORCE)
# set(TORCH_CUDA_ARCH_LIST "8.0")

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "Please set CMAKE_CUDA_ARCHITECTURES")
endif()

# check the CMAKE_CUDA_ARCHITECTURES is not a list
if (CMAKE_CUDA_ARCHITECTURES MATCHES ";")
  message(FATAL_ERROR "Please set CMAKE_CUDA_ARCHITECTURES as a single value")
endif()

# convert CMAKE_CUDA_ARCHITECTURES to TORCH_CUDA_ARCH_LIST
set(TORCH_CUDA_ARCH_LIST "")
if (CMAKE_CUDA_ARCHITECTURES STREQUAL "80")
  set(TORCH_CUDA_ARCH_LIST "8.0")
elseif (CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
  set(TORCH_CUDA_ARCH_LIST "8.6")
elseif (CMAKE_CUDA_ARCHITECTURES STREQUAL "89")
  set(TORCH_CUDA_ARCH_LIST "8.9")
elseif (CMAKE_CUDA_ARCHITECTURES STREQUAL "90")
  set(TORCH_CUDA_ARCH_LIST "9.0")
elseif (CMAKE_CUDA_ARCHITECTURES STREQUAL "90a")
  set(TORCH_CUDA_ARCH_LIST "9.0a")
else()
  message(FATAL_ERROR "Unsupported CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
endif()


#################### PROLOGUE END ####################

set(PROJ_NAME "custom_api_test")
project(${PROJ_NAME} LANGUAGES CXX CUDA)

###### CUDA LANGUAGE MUST BE ENABLED BEFORE TORCH ######

set(CMAKE_UTIL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(${CMAKE_UTIL_DIR}/utilsFuncs.cmake)

locate_file_dir(hopper FA3_ROOT_CONTAINING_DIR)
set(FA3_ROOT_DIR ${FA3_ROOT_CONTAINING_DIR}/hopper)
message(STATUS "Hopper root dir: ${FA3_ROOT_DIR}")

locate_file_dir(.venv FA_VENV_CONTAINING_DIR)
set(FA_VENV_DIR ${FA_VENV_CONTAINING_DIR}/.venv)
message(STATUS "FA VENV dir: ${FA_VENV_DIR}")

set(Torch_DIR "${FA_VENV_DIR}/lib/python3.11/site-packages/torch/share/cmake/Torch")
find_package(Torch CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development)

#### FA csrc for CUTLASS

locate_file_dir(csrc FA_CSRC_CONTAINING_DIR)
set(FA_CSRC_DIR ${FA_CSRC_CONTAINING_DIR}/csrc)
set(CUTLASS_INCLUDE_DIR ${FA_CSRC_DIR}/cutlass/include)
assert_dir_exists(${CUTLASS_INCLUDE_DIR})
message(STATUS "CUTLASS include dir: ${CUTLASS_INCLUDE_DIR}")

set(CUSTOM_API_DIR ${CMAKE_CURRENT_SOURCE_DIR})

#### FA3 library

function(add_fa3_lib SOURCEFILE LIBNAME)
  # target name is the source file name without extension
  add_library(${LIBNAME} SHARED ${SOURCEFILE})
  target_compile_options(${LIBNAME} PRIVATE --expt-relaxed-constexpr)
  target_compile_options(${LIBNAME} PRIVATE --expt-extended-lambda)
  target_compile_options(${LIBNAME} PRIVATE --use_fast_math)
  target_include_directories(${LIBNAME} PRIVATE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(${LIBNAME} PRIVATE ${TORCH_LIBRARIES})
  target_include_directories(${LIBNAME} PRIVATE ${CUTLASS_INCLUDE_DIR})
  target_include_directories(${LIBNAME} PRIVATE ${FA3_ROOT_DIR})
  target_include_directories(${LIBNAME} PRIVATE ${CUSTOM_API_DIR})
endfunction()

add_fa3_lib(${FA3_ROOT_DIR}/flash_prepare_scheduler.cu fa3_prepare_scheduler)

function(add_single_source_executable SOURCEFILE)
  # target name is the source file name without extension
  get_filename_component(TARGET_NAME ${SOURCEFILE} NAME_WE)
  add_executable(${TARGET_NAME} ${SOURCEFILE})
  target_compile_options(${TARGET_NAME} PRIVATE --expt-relaxed-constexpr)
  target_compile_options(${TARGET_NAME} PRIVATE --expt-extended-lambda)
  target_compile_options(${TARGET_NAME} PRIVATE --use_fast_math)
  target_include_directories(${TARGET_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(${TARGET_NAME} PRIVATE ${TORCH_LIBRARIES})
  target_include_directories(${TARGET_NAME} PRIVATE ${CUTLASS_INCLUDE_DIR})
  target_include_directories(${TARGET_NAME} PRIVATE ${FA3_ROOT_DIR})
  target_include_directories(${TARGET_NAME} PRIVATE ${CUSTOM_API_DIR})
  target_link_libraries(${TARGET_NAME} PRIVATE fa3_prepare_scheduler)
endfunction()

add_single_source_executable(test.cu)