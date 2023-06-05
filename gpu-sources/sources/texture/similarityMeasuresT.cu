#include "similarityMeasuresT.cuh"

__global__ void calculateTextureComponents(uint8_t* image, uint8_t* background, float* result, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;
        uint8_t vector = ~(image[index] ^ background[index]);
        result[index] = __popc(vector) / 8.0f;
    }
}