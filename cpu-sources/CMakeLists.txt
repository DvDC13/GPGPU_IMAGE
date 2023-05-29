cmake_minimum_required(VERSION 3.22.1)
project(gpgpu)

set(CMAKE_CXX_STANDARD 20)
# disable warnings for deprecated enum conversion in OpenCV
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-deprecated-enum-enum-conversion -g")

#TODO: rm png
find_package(PNG REQUIRED)
find_package(OpenCV REQUIRED)

#TODO: rm png
include_directories(${PNG_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sources)

add_library(cpu_lib STATIC)
target_include_directories(cpu_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/includes
                                          ${CMAKE_CURRENT_SOURCE_DIR}/includes/color
                                          ${CMAKE_CURRENT_SOURCE_DIR}/includes/utility
                                          ${CMAKE_CURRENT_SOURCE_DIR}/includes/texture)

add_subdirectory(sources)

add_executable(gpgpu main.cpp)

target_link_libraries(gpgpu cpu_lib ${PNG_LIBRARY} ${OpenCV_LIBS})

set(DATASET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../dataset)

target_compile_definitions(gpgpu PRIVATE DATASET_DIR="${DATASET_DIR}")