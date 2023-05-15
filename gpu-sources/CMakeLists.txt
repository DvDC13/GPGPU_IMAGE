cmake_minimum_required(VERSION 3.22.1)
project(GPGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(gpu_lib STATIC)
target_include_directories(gpu_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/includes)

add_subdirectory(sources)

add_executable(gpgpu main.cu)

target_link_libraries(gpgpu PRIVATE gpu_lib)