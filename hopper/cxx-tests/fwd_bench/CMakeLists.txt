cmake_minimum_required(VERSION 3.26)

set(PROJ_NAME "template")
set(CMAKE_UTILS_DIR "${CMAKE_CURRENT_LIST_DIR}/../cmake")

############################################################
############### DO NOT EDIT BELOW THIS LINE ################

# include the prologue file BEFORE calling project() !!!
include(${CMAKE_UTILS_DIR}/cmakePrologue.cmake)
include(${CMAKE_UTILS_DIR}/utilFuncs.cmake)
project(${PROJ_NAME} LANGUAGES CXX CUDA)
#
# flashVenv.cmake provides "add_single_source_executable" function
#
include(${CMAKE_UTILS_DIR}/flashVenv.cmake)

############### DO NOT EDIT ABOVE THIS LINE ################
############################################################

### csv2

include(FetchContent)
FetchContent_Declare(
    csv2
    GIT_REPOSITORY https://github.com/p-ranav/csv2.git
    GIT_TAG v0.1
    # GIT_CONFIG "http.proxy=127.0.0.1:7890" "https.proxy=127.0.0.1:7890"
)
FetchContent_MakeAvailable(csv2)

function(link_to_csv2 TARGET_NAME)
  target_include_directories(${TARGET_NAME} PRIVATE ${csv2_SOURCE_DIR}/include)
  target_link_libraries(${TARGET_NAME} PRIVATE csv2::csv2)
endfunction()

### argparse

FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v3.1
    # GIT_CONFIG "http.proxy=127.0.0.1:7890" "https.proxy=127.0.0.1:7890"
)
FetchContent_MakeAvailable(argparse)

function(link_to_argparse TARGET_NAME)
  target_include_directories(${TARGET_NAME} PRIVATE ${argparse_SOURCE_DIR}/include)
  target_link_libraries(${TARGET_NAME} PRIVATE argparse::argparse)
endfunction()

add_single_source_executable(fp8-fwd-bench.cu)
link_to_csv2(fp8-fwd-bench)
link_to_argparse(fp8-fwd-bench)

add_single_source_executable(fp16-fwd-bench.cu)
link_to_csv2(fp16-fwd-bench)
link_to_argparse(fp16-fwd-bench)


add_single_source_executable(fp8-fwd-bench-orig.cu)
link_to_csv2(fp8-fwd-bench-orig)
link_to_argparse(fp8-fwd-bench-orig)

add_single_source_executable(fp16-fwd-bench-orig.cu)
link_to_csv2(fp16-fwd-bench-orig)
link_to_argparse(fp16-fwd-bench-orig)
