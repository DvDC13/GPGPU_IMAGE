#include "similarityMeasuresT.cuh"

__global__ void calculateTextureComponents(uint8_t* image, uint8_t* background, float* result, size_t batch_size, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < batch_size)
    {
        int index_frame = x + y * width + z * width * height;
        int index_background = x + y * width;
        uint8_t vector = ~(image[index_frame] ^ background[index_background]);
        result[index_frame] = __popc(vector) / 8.0f;
    }
}