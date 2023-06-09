#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "image.cuh"

__device__ float getGrayscale(const Pixel pixel);

__global__ void calculateBitVectorBackground(const Pixel* imageData, uint8_t* bitVectorData, int width, int height);

__global__ void calculateBitVector(const Pixel* imageData, uint8_t* bitVectorData, size_t batch_index, size_t batch_size, int width, int height);