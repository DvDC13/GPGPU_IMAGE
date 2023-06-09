#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "image.cuh"

__global__ void calculateTextureComponents(uint8_t* image, uint8_t* background, float* result, size_t batch_size, int width, int height);