cmake_minimum_required(VERSION 3.22.1)
project(gpgpu)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -g")

find_package(PNG REQUIRED)

include_directories(${PNG_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes)

add_library(cpu_lib STATIC)
target_include_directories(cpu_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/includes)

add_subdirectory(sources)

add_executable(gpgpu main.cpp)

target_link_libraries(gpgpu cpu_lib ${PNG_LIBRARY})