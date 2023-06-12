#pragma once

__device__ char* get_3d(char* data, size_t x, size_t y, size_t z, size_t pitch,
                        size_t height, size_t elm_size);
