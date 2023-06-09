#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "image.cuh"

__global__ void calculateChoquetMask(const Pixel* colorComponents, const float* textureComponents, Bit* result, size_t batch_index, size_t batch_size, int width, int height);