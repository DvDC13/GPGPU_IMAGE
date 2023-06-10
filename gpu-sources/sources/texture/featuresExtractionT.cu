#include "featuresExtractionT.cuh"

__device__ float getGrayscale(const Pixel pixel)
{
    return 0.299f * pixel[0] + 0.587f * pixel[1] + 0.114f * pixel[2];
}

__global__ void calculateBitVectorBackground(const Pixel* imageData, uint8_t*
        bitVectorData, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;
        uint8_t vector = 0;

        float gray = getGrayscale(imageData[index]);

        auto isBorder = [&](int dx, int dy) {
            if (x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height) {
                return getGrayscale(imageData[(y + dy) * width + (x + dx)]);
            }
            else {
                return 255.0f;
            }
        };

        vector = (vector << 1) | (isBorder(-1, -1) < gray);
        vector = (vector << 1) | (isBorder(0, -1) < gray);
        vector = (vector << 1) | (isBorder(1, -1) < gray);
        vector = (vector << 1) | (isBorder(1, 0) < gray);
        vector = (vector << 1) | (isBorder(1, 1) < gray);
        vector = (vector << 1) | (isBorder(0, 1) < gray);
        vector = (vector << 1) | (isBorder(-1, 1) < gray);
        vector = (vector << 1) | (isBorder(-1, 0) < gray);

        bitVectorData[index] = vector;
    }
}

__global__ void calculateBitVector(const Pixel* imageData, uint8_t* bitVectorData, size_t batch_size, int width, int height, size_t imagePitch, size_t bitVecPitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < batch_size)
    {
        //int indexImage = z * width * height + y * width + x;
        int indexImage = x * sizeof(Pixel) + y * imagePitch + z * imagePitch * height;
        int indexBitVec = x * sizeof(uint8_t) + y * bitVecPitch + z * bitVecPitch * height;

        uint8_t vector = 0;

        Pixel* pixelCenter = (Pixel *) (((char *) imageData) + indexImage);
        float gray = getGrayscale(*pixelCenter);

        auto isBorder = [&](int dx, int dy) {
            if (x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height) {
                //return getGrayscale(imageData[(y + dy) * width + (x + dx)]);
                int subindexImage = (x + dx) * sizeof(Pixel) + (y + dy) * imagePitch + z * imagePitch * height;
                Pixel* pixel = (Pixel *) (((char *) imageData) + subindexImage);
                return getGrayscale(*pixel);
            }
            else {
                return 255.0f;
            }
        };

        vector = (vector << 1) | (isBorder(-1, -1) < gray);
        vector = (vector << 1) | (isBorder(0, -1) < gray);
        vector = (vector << 1) | (isBorder(1, -1) < gray);
        vector = (vector << 1) | (isBorder(1, 0) < gray);
        vector = (vector << 1) | (isBorder(1, 1) < gray);
        vector = (vector << 1) | (isBorder(0, 1) < gray);
        vector = (vector << 1) | (isBorder(-1, 1) < gray);
        vector = (vector << 1) | (isBorder(-1, 0) < gray);

        bitVectorData[indexBitVec] = vector;
    }
}
