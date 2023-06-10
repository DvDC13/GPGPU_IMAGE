#include "similarityMeasuresT.cuh"

__global__ void calculateTextureComponents(uint8_t* image, uint8_t* background,
        float* result, size_t batch_size, int width, int height, size_t bitVecPitch, size_t texturePitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < batch_size)
    {
        //int index_frame = x + y * width + z * width * height;
        int index_frame = x + y * bitVecPitch / sizeof(uint8_t) + z * bitVecPitch / sizeof(uint8_t) * height;
        int index_result = x + y * texturePitch / sizeof(float) + z *
            texturePitch / sizeof(float) * height;
        int index_background = x + y * width;
        uint8_t vector = ~(image[index_frame] ^ background[index_background]);
        result[index_result] = __popc(vector) / 8.0f;
    }
}
