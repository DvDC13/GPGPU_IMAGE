cmake_minimum_required(VERSION 3.22.1)
project(gpgpu)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -fsanitize=address -O3")
set(CMAKE_EXE_LINKER_FLAGS "-fsanitize=address -O3")


find_package(PNG REQUIRED)

include_directories(${PNG_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sources)

add_library(cpu_lib STATIC)
target_include_directories(cpu_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/includes
                                          ${CMAKE_CURRENT_SOURCE_DIR}/includes/color
                                          ${CMAKE_CURRENT_SOURCE_DIR}/includes/utility
                                          ${CMAKE_CURRENT_SOURCE_DIR}/includes/texture)

add_subdirectory(sources)

add_executable(gpgpu main.cpp)

target_link_libraries(gpgpu cpu_lib ${PNG_LIBRARY})

set(DATASET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../dataset)

target_compile_definitions(gpgpu PRIVATE DATASET_DIR="${DATASET_DIR}")