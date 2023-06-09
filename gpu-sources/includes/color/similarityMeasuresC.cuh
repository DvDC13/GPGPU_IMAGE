#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "image.cuh"

__global__ void calculateSimilarityMeasures(const Pixel* imageData, const Pixel* backgroundData, std::array<float, 2>* result, size_t batch_size, int width, int height);