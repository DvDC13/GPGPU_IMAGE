cmake_minimum_required(VERSION 3.22.1)
project(gpgpu)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -g")

find_package(PNG REQUIRED)

include_directories(${PNG_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sources)

add_library(cpu_lib STATIC)
target_include_directories(cpu_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/includes
                                          ${CMAKE_CURRENT_SOURCE_DIR}/includes/color)

add_subdirectory(sources/utility)

add_executable(gpgpu main.cc)

target_link_libraries(gpgpu cpu_lib ${PNG_LIBRARY})

set(DATASET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../dataset)

target_compile_definitions(gpgpu PRIVATE DATASET_DIR="${DATASET_DIR}")