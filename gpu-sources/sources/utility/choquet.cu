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

__global__ void calculateChoquetMask(const Pixel* colorComponents,
                                         const float* textureComponents,
                                         Bit* result, size_t batch_index, size_t batch_size, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < batch_size)
    {
        int index = (z + batch_index) * width * height + y * width + x;

        float indicators[3];
        indicators[0] = colorComponents[index][0];
        indicators[1] = colorComponents[index][1];
        indicators[2] = textureComponents[index];

        sort3<float>(indicators);

        float choquet = indicators[0] * 0.1f + indicators[1] * 0.3f + indicators[2] * 0.6f;

        result[index] = choquet > 0.67f ? false : true;
    }
}