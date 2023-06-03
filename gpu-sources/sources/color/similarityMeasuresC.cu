#include "similarityMeasuresC.cuh"

__global__ void calculateSimilarityMeasures(const Pixel* imageData, const Pixel* backgroundData, Pixel* result, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;

        float red_image = imageData[index][0];
        float green_image = imageData[index][1];
        float blue_image = imageData[index][2];

        float red_background = backgroundData[index][0];
        float green_background = backgroundData[index][1];
        float blue_background = backgroundData[index][2];

        float similarity_red = fminf(red_image, red_background) / fmaxf(red_image, red_background);
        float similarity_green = fminf(green_image, green_background) / fmaxf(green_image, green_background);
        float similarity_blue = fminf(blue_image, blue_background) / fmaxf(blue_image, blue_background);

        result[index][0] = similarity_red;
        result[index][1] = similarity_green;
        result[index][2] = similarity_blue;
    }
}