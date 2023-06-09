#include "similarityMeasuresC.cuh"

__global__ void calculateSimilarityMeasures(const Pixel* imageData, const Pixel* backgroundData, Pixel* result, size_t batch_index, size_t batch_size, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < batch_size)
    {
        size_t index_frame = x + y * width + (z +  batch_index) * width * height;
        size_t index_background = x + y * width;
        
        float red_image = imageData[index_frame][0];
        float green_image = imageData[index_frame][1];
        float blue_image = imageData[index_frame][2];

        float red_background = backgroundData[index_background][0];
        float green_background = backgroundData[index_background][1];
        float blue_background = backgroundData[index_background][2];

        float similarity_red = fminf(red_image, red_background) / fmaxf(red_image, red_background);
        float similarity_green = fminf(green_image, green_background) / fmaxf(green_image, green_background);
        float similarity_blue = fminf(blue_image, blue_background) / fmaxf(blue_image, blue_background);

        result[index_frame][0] = similarity_red;
        result[index_frame][1] = similarity_green;
        result[index_frame][2] = similarity_blue;
    }
}