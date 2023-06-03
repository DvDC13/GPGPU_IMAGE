#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "image.h"

__global__ void calculateChoquetIntegral(const Pixel* colorComponents, const float* textureComponents, float* result, int width, int height);

__global__ void calculateMask(const float* choquetIntegral, Bit* result, int width, int height, float threshold);