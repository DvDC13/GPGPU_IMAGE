#include "choquet.cuh"

template <typename T>
__device__ void swap(T& a, T& b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

template <typename T, std::size_t N>
__device__ void sort3(T (&arr)[N])
{
    if (N < 3)
        return;
    if (arr[0] > arr[1])
        swap(arr[0], arr[1]);
    if (arr[1] > arr[2])
        swap(arr[1], arr[2]);
    if (arr[0] > arr[1])
        swap(arr[0], arr[1]);
}

__device__ char* get_3d(char* data, size_t x, size_t y, size_t z, size_t pitch,
                        size_t height, size_t elm_size)
{
    return data + y * pitch + x * elm_size + z * pitch * height;
}

__global__ void
calculateChoquetMask(const std::array<float, 2>* colorComponents,
                     const float* textureComponents, Bit* result,
                     size_t batch_size, int width, int height,
                     size_t colorPitch, size_t texturePitch, size_t masksPitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < batch_size)
    {
        int colorIndex = x + y * colorPitch / sizeof(std::array<float, 2>)
            + z * colorPitch / sizeof(std::array<float, 2>) * height;
        int textureIndex = x + y * texturePitch / sizeof(float)
            + z * texturePitch / sizeof(float) * height;
        int resultIndex = x + y * masksPitch / sizeof(Bit)
            + z * masksPitch / sizeof(Bit) * height;

        std::array<float, 3>* color_ptr = (std::array<float, 3>*)get_3d(
            (char*)colorComponents, x, y, z, colorPitch, height,
            sizeof(std::array<float, 2>));
        float* texture_ptr =
            (float*)get_3d((char*)textureComponents, x, y, z, texturePitch,
                           height, sizeof(float));
        Bit* result_ptr = (Bit*)get_3d((char*)result, x, y, z, masksPitch,
                                       height, sizeof(Bit));

        float indicators[3];
        indicators[0] = (*color_ptr)[0];
        indicators[1] = (*color_ptr)[1];
        indicators[2] = *texture_ptr;

        sort3<float>(indicators);

        float choquet =
            indicators[0] * 0.1f + indicators[1] * 0.3f + indicators[2] * 0.6f;

        *result_ptr = choquet > 0.67f ? false : true;
    }
}
