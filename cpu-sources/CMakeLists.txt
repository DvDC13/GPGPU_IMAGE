cmake_minimum_required(VERSION 3.22.1)
project(gpgpu)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -g")

find_package(PNG REQUIRED)

# Copy files to build directory
# Define the source and destination directories
set(source_dir "dataset/frames")
set(destination_dir "${CMAKE_BINARY_DIR}/dataset/frames")

# Get a list of source files
file(GLOB source_files "${source_dir}/*.png")

# Copy files to the build directory
foreach(source_file ${source_files})
    get_filename_component(file_name ${source_file} NAME)
    configure_file(${source_file} ${destination_dir}/${file_name} COPYONLY)
endforeach()

include_directories(${PNG_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes)

add_library(cpu_lib STATIC)
target_include_directories(cpu_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/includes)

add_subdirectory(sources)

add_executable(gpgpu main.cpp)

target_link_libraries(gpgpu cpu_lib ${PNG_LIBRARY})