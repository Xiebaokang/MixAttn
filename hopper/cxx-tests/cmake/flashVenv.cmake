set(USE_SYSTEM_NVTX ON)

locate_file_dir(hopper FA3_ROOT_CONTAINING_DIR)
set(FA3_ROOT_DIR ${FA3_ROOT_CONTAINING_DIR}/hopper)
message(STATUS "Hopper root dir: ${FA3_ROOT_DIR}")

if (DEFINED FA_VENV_DIR)
  set(FA_VENV_DIR ${FA_VENV_DIR} CACHE PATH "Python environment with PyTorch")
elseif (DEFINED ENV{VIRTUAL_ENV})
  set(FA_VENV_DIR $ENV{VIRTUAL_ENV})
elseif (DEFINED ENV{CONDA_PREFIX})
  set(FA_VENV_DIR $ENV{CONDA_PREFIX})
else()
  locate_file_dir(.venv FA_VENV_CONTAINING_DIR)
  set(FA_VENV_DIR ${FA_VENV_CONTAINING_DIR}/.venv)
endif()
assert_dir_exists(${FA_VENV_DIR})
message(STATUS "FA Python env dir: ${FA_VENV_DIR}")

list(APPEND PYTHON_VER_CANDIDATES 3.10 3.11 3.12 3.13)
foreach(PYTHON_VER ${PYTHON_VER_CANDIDATES})
  set(VENV_PYTHON_LIB_DIR ${FA_VENV_DIR}/lib/python${PYTHON_VER})
  if (EXISTS ${VENV_PYTHON_LIB_DIR})
    set(PYTHON_LIB_DIR ${VENV_PYTHON_LIB_DIR})
    message(STATUS "Found Python lib dir: ${PYTHON_LIB_DIR}")
  endif()
endforeach()


set(Torch_DIR "${PYTHON_LIB_DIR}/site-packages/torch/share/cmake/Torch")
assert_dir_exists(${Torch_DIR})
find_package(Torch CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development)

#### CUTLASS

locate_file_dir(cutlass CUTLASS_CONTAINING_DIR)
set(CUTLASS_ROOT_DIR ${CUTLASS_CONTAINING_DIR}/cutlass)
set(CUTLASS_INCLUDE_DIR ${CUTLASS_ROOT_DIR}/include)
message(STATUS "CUTLASS include dir: ${CUTLASS_INCLUDE_DIR}")


#### CUSTOM API DIR

locate_file_dir(custom_api CUSTOM_API_CONTAINING_DIR)
set(CUSTOM_API_DIR ${CUSTOM_API_CONTAINING_DIR}/custom_api)
assert_dir_exists(${CUSTOM_API_DIR})
message(STATUS "Custom API dir: ${CUSTOM_API_DIR}")

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
  if(ARGC GREATER 1)
    set(REGISTER_USAGE_LEVEL ${ARGV1})
  else()
    set(REGISTER_USAGE_LEVEL 10)
  endif()
  target_compile_options(
    ${TARGET_NAME} PRIVATE -Xptxas --register-usage-level=${REGISTER_USAGE_LEVEL}
  )
  target_include_directories(${TARGET_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(${TARGET_NAME} PRIVATE ${TORCH_LIBRARIES})
  target_include_directories(${TARGET_NAME} PRIVATE ${CUTLASS_INCLUDE_DIR})
  target_include_directories(${TARGET_NAME} PRIVATE ${FA3_ROOT_DIR})
  target_include_directories(${TARGET_NAME} PRIVATE ${CUSTOM_API_DIR})
  target_link_libraries(${TARGET_NAME} PRIVATE fa3_prepare_scheduler)
endfunction()

function(compile_to_ptx_ FILENAME)
  cmake_parse_arguments(PTX "" "" "INCLUDE_DIRS" ${ARGN})
  if(NOT CMAKE_CUDA_COMPILER)
      message(FATAL_ERROR "CUDA compiler not found. Ensure CUDA is enabled in the project.")
  endif()
  get_filename_component(FILE_BASENAME ${FILENAME} NAME_WE)
  set(PTX_OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${FILE_BASENAME}.ptx")
  if(IS_ABSOLUTE ${FILENAME})
      set(CUDA_SOURCE_FILE ${FILENAME})
  else()
      set(CUDA_SOURCE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
  endif()
  set(INCLUDE_FLAGS "")
  foreach(DIR ${PTX_INCLUDE_DIRS})
      list(APPEND INCLUDE_FLAGS "-I${DIR}")
  endforeach()
  add_custom_command(
      OUTPUT ${PTX_OUTPUT_FILE}
      COMMAND ${CMAKE_CUDA_COMPILER} -ptx ${CUDA_SOURCE_FILE} -o ${PTX_OUTPUT_FILE} --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math --gpu-architecture=sm_90a ${INCLUDE_FLAGS}
      DEPENDS ${CUDA_SOURCE_FILE}
      COMMENT "Generating PTX file ${PTX_OUTPUT_FILE} from ${CUDA_SOURCE_FILE}"
      VERBATIM
  )
  add_custom_target(generate_ptx_${FILE_BASENAME} ALL
      DEPENDS ${PTX_OUTPUT_FILE}
  )
endfunction()

function(compile_to_ptx FILENAME)
  # target_include_directories(${TARGET_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
  # target_include_directories(${TARGET_NAME} PRIVATE ${CUTLASS_INCLUDE_DIR})
  # target_include_directories(${TARGET_NAME} PRIVATE ${FA3_ROOT_DIR})
  # target_include_directories(${TARGET_NAME} PRIVATE ${CUSTOM_API_DIR})

  #include the above directories in the function
  list(APPEND INCLUDE_DIRS_ ${TORCH_INCLUDE_DIRS} ${CUTLASS_INCLUDE_DIR} ${FA3_ROOT_DIR} ${CUSTOM_API_DIR})
  compile_to_ptx_(${FILENAME} INCLUDE_DIRS ${INCLUDE_DIRS_})
endfunction()
