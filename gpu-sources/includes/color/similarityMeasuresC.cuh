#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "image.h"

__global__ void calculateSimilarityMeasures(const Pixel* imageData, const Pixel* backgroundData, Pixel* result, int width, int height);