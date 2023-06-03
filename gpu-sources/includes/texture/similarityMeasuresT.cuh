#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "image.h"

__global__ void calculateTextureComponents(uint8_t* image, uint8_t* background, float* result, int width, int height);