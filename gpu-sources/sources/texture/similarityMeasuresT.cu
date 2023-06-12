#include "similarityMeasuresT.cuh"

__device__ char* get_3d(char* data, size_t x, size_t y, size_t z, size_t pitch,
                        size_t height, size_t elm_size)
{
    return data + y * pitch + x * elm_size + z * pitch * height;
}

__global__ void calculateTextureComponents(uint8_t* image, uint8_t* background,
                                           float* result, size_t batch_size,
                                           int width, int height,
                                           size_t bitVecPitch,
                                           size_t texturePitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < batch_size)
    {
        // int index_frame = x + y * width + z * width * height;
        int index_frame = x + y * bitVecPitch / sizeof(uint8_t)
            + z * bitVecPitch / sizeof(uint8_t) * height;
        int index_result = x + y * texturePitch / sizeof(float)
            + z * texturePitch / sizeof(float) * height;
        int index_background = x + y * width;

        uint8_t* frame = (uint8_t*)get_3d((char*)image, x, y, z, bitVecPitch,
                                          height, sizeof(uint8_t));
        float* result_ptr = (float*)get_3d((char*)result, x, y, z, texturePitch,
                                        height, sizeof(float));
        uint8_t vector = ~(*frame ^ background[index_background]);
        *result_ptr = __popc(vector) / 8.0f;
    }
}
