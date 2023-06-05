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

__global__ void calculateChoquetIntegral(const Pixel* colorComponents, const float* textureComponents, float* result, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;

        float indicators[3];
        indicators[0] = colorComponents[index][0];
        indicators[1] = colorComponents[index][1];
        indicators[2] = colorComponents[index][2];

        sort3<float>(indicators);

        result[index] = indicators[0] * 0.1f + indicators[1] * 0.3f + indicators[2] * 0.6f;
    }
}

__global__ void calculateMask(const float* choquetIntegral, Bit* result, int width, int height, float threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;
        result[index] = choquetIntegral[index] > threshold ? false : true;
    }
}