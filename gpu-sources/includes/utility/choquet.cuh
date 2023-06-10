#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "image.cuh"

__global__ void calculateChoquetMask(const std::array<float, 2>*
        colorComponents, const float* textureComponents, Bit* result, size_t
        batch_size, int width, int height, size_t colorPitch, size_t texturePitch, size_t masksPitch);
