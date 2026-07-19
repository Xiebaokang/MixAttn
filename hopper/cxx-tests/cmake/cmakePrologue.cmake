
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