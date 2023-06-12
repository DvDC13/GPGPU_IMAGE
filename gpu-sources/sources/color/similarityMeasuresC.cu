#include "similarityMeasuresC.cuh"
#include "index.cuh"


__global__ void calculateSimilarityMeasures(const Pixel* imageData,
                                            const Pixel* backgroundData,
                                            std::array<float, 2>* result,
                                            size_t batch_size, int width,
                                            int height, size_t imagePitch,
                                            size_t colorPitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < batch_size)
    {
        // size_t index_frame = x + y * width + z * width * height;
        size_t index_frame = x + y * imagePitch / sizeof(Pixel)
            + z * imagePitch / sizeof(Pixel) * height;

        size_t index_background = x + y * width;

        Pixel* frame = (Pixel*)get_3d((char*)imageData, x, y, z, imagePitch,
                                      height, sizeof(Pixel));

        float red_image = (*frame)[0];
        float green_image = (*frame)[1];

        float red_background = backgroundData[index_background][0];
        float green_background = backgroundData[index_background][1];

        float similarity_red =
            fminf(red_image, red_background) / fmaxf(red_image, red_background);
        float similarity_green = fminf(green_image, green_background)
            / fmaxf(green_image, green_background);

        size_t result_index = x + y * colorPitch / sizeof(std::array<float, 2>)
            + z * colorPitch / sizeof(std::array<float, 2>) * height;

        std::array<float, 2>* result_ptr = (std::array<float, 2>*) get_3d(
            (char*)result, x, y, z, colorPitch, height,
            sizeof(std::array<float, 2>));

        (*result_ptr)[0] = similarity_red;
        (*result_ptr)[1] = similarity_green;
    }
}
