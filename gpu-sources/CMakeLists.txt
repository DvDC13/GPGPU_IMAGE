cmake_minimum_required(VERSION 3.22.1)
project(GPGPU LANGUAGES CUDA CXX)

find_package(PNG REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter -Wno-unused-variable -O3")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -gencode arch=compute_75,code=sm_75 --expt-relaxed-constexpr --use_fast_math -O3")

include_directories(${PNG_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes
                    ${CMAKE_CURRENT_SOURCE_DIR}/includes/utility
                    ${CMAKE_CURRENT_SOURCE_DIR}/includes/texture
                    ${CMAKE_CURRENT_SOURCE_DIR}/includes/color)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sources)

add_executable(gpgpu main.cu sources/utility/image.cu
                            sources/utility/image.cuhxx
                            sources/utility/png_utility.cu
                            sources/utility/choquet.cu
                            sources/texture/featuresExtractionT.cu
                            sources/texture/similarityMeasuresT.cu
                            sources/color/similarityMeasuresC.cu)

target_link_libraries(gpgpu PRIVATE ${PNG_LIBRARY})
