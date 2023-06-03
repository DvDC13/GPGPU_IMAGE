#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "image.h"

__device__ float getGrayscale(const Pixel pixel);

__global__ void calculateBitVector(const Pixel* imageData, uint8_t* bitVectorData, int width, int height);