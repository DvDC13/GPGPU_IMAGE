#include <algorithm>
#include <filesystem>

#include "wrapper_gpu.cuh"
#include "similarityMeasuresC.cuh"
#include "featuresExtractionT.cuh"
#include "similarityMeasuresT.cuh"
#include "choquet.cuh"

#include "image.h"

shared_bit_vector getBitVector(shared_image image)
{
    int width = image->get_width();
    int height = image->get_height();
    int size = width * height;

    // Allocate device memory
    Pixel* deviceImageData;
    uint8_t* deviceBitVectorData;
    cudaXMalloc((void**)&deviceImageData, size * sizeof(Pixel));
    cudaXMalloc((void**)&deviceBitVectorData, size * sizeof(uint8_t));

    // Copy image data from host to device
    cudaXMemcpy(deviceImageData, image->get_data().data(), size * sizeof(Pixel), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    calculateBitVector<<<gridSize, blockSize>>>(deviceImageData, deviceBitVectorData, width, height);
    cudaDeviceSynchronize();

    // Copy bit vector data from device to host
    std::vector<uint8_t> hostBitVectorData(size);
    cudaXMemcpy(hostBitVectorData.data(), deviceBitVectorData, size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaXFree(deviceImageData);
    cudaXFree(deviceBitVectorData);

    // Create a shared_bit_vector from the host data
    shared_bit_vector result = std::make_shared<Image<uint8_t>>(width, height);
    result->set_data(hostBitVectorData);

    return result;
}

shared_image getColorImage(shared_image image, shared_image background)
{
    int width = image->get_width();
    int height = image->get_height();
    int size = width * height;

    // Allocate device memory
    Pixel* deviceImageData;
    Pixel* deviceBackgroundData;
    Pixel* deviceColorImageData;

    cudaXMalloc((void**)&deviceImageData, size * sizeof(Pixel));
    cudaXMalloc((void**)&deviceBackgroundData, size * sizeof(Pixel));
    cudaXMalloc((void**)&deviceColorImageData, size * sizeof(Pixel));

    // Copy image data from host to device
    cudaXMemcpy(deviceImageData, image->get_data().data(), size * sizeof(Pixel), cudaMemcpyHostToDevice);
    cudaXMemcpy(deviceBackgroundData, background->get_data().data(), size * sizeof(Pixel), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    calculateSimilarityMeasures<<<gridSize, blockSize>>>(deviceImageData, deviceBackgroundData, deviceColorImageData, width, height);
    cudaDeviceSynchronize();

    // Copy color image data from device to host
    std::vector<Pixel> hostColorImageData(size);
    cudaXMemcpy(hostColorImageData.data(), deviceColorImageData, size * sizeof(Pixel), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaXFree(deviceImageData);
    cudaXFree(deviceBackgroundData);
    cudaXFree(deviceColorImageData);

    // Create a shared_image from the host data
    shared_image result = std::make_shared<Image<Pixel>>(width, height);
    result->set_data(hostColorImageData);

    return result;
}

shared_float_vector getTextureComponents(shared_bit_vector image, shared_bit_vector background)
{
    int width = image->get_width();
    int height = image->get_height();
    int size = width * height;

    // Allocate device memory
    uint8_t* deviceImageData;
    uint8_t* deviceBackgroundData;
    float* deviceTextureComponentsData;

    cudaXMalloc((void**)&deviceImageData, size * sizeof(uint8_t));
    cudaXMalloc((void**)&deviceBackgroundData, size * sizeof(uint8_t));
    cudaXMalloc((void**)&deviceTextureComponentsData, size * sizeof(float));

    // Copy image data from host to device
    cudaXMemcpy(deviceImageData, image->get_data().data(), size * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaXMemcpy(deviceBackgroundData, background->get_data().data(), size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    calculateTextureComponents<<<gridSize, blockSize>>>(deviceImageData, deviceBackgroundData, deviceTextureComponentsData, width, height);
    cudaDeviceSynchronize();

    // Copy texture components data from device to host
    std::vector<float> hostTextureComponentsData(size);
    cudaXMemcpy(hostTextureComponentsData.data(), deviceTextureComponentsData, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaXFree(deviceImageData);
    cudaXFree(deviceBackgroundData);
    cudaXFree(deviceTextureComponentsData);

    // Create a shared_float_vector from the host data
    shared_float_vector result = std::make_shared<Image<float>>(width, height);
    result->set_data(hostTextureComponentsData);

    return result;
}

shared_float_vector computeChoquetIntegral(shared_float_vector textureComponents, shared_image colorComponents)
{
    int width = textureComponents->get_width();
    int height = textureComponents->get_height();
    int size = width * height;

    // Allocate device memory
    float* deviceTextureComponentsData;
    Pixel* deviceColorComponentsData;
    float* deviceChoquetIntegralData;

    cudaXMalloc((void**)&deviceTextureComponentsData, size * sizeof(float));
    cudaXMalloc((void**)&deviceColorComponentsData, size * sizeof(Pixel));
    cudaXMalloc((void**)&deviceChoquetIntegralData, size * sizeof(float));

    // Copy image data from host to device
    cudaXMemcpy(deviceTextureComponentsData, textureComponents->get_data().data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaXMemcpy(deviceColorComponentsData, colorComponents->get_data().data(), size * sizeof(Pixel), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    calculateChoquetIntegral<<<gridSize, blockSize>>>(deviceColorComponentsData, deviceTextureComponentsData, deviceChoquetIntegralData, width, height);
    cudaDeviceSynchronize();

    // Copy choquet integral data from device to host
    std::vector<float> hostChoquetIntegralData(size);
    cudaXMemcpy(hostChoquetIntegralData.data(), deviceChoquetIntegralData, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(deviceTextureComponentsData);
    cudaFree(deviceColorComponentsData);
    cudaFree(deviceChoquetIntegralData);

    // Create a shared_float_vector from the host data
    shared_float_vector result = std::make_shared<Image<float>>(width, height);
    result->set_data(hostChoquetIntegralData);

    return result;
}

shared_mask getMaskResult(shared_float_vector choquetIntegral, float threshold)
{
    int width = choquetIntegral->get_width();
    int height = choquetIntegral->get_height();
    int size = width * height;

    // Allocate device memory
    float* deviceChoquetIntegralData;
    Bit* deviceMaskData;

    cudaXMalloc((void**)&deviceChoquetIntegralData, size * sizeof(float));
    cudaXMalloc((void**)&deviceMaskData, size * sizeof(Bit));

    // Copy image data from host to device
    cudaXMemcpy(deviceChoquetIntegralData, choquetIntegral->get_data().data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    calculateMask<<<gridSize, blockSize>>>(deviceChoquetIntegralData, deviceMaskData, width, height, threshold);
    cudaDeviceSynchronize();

    // Copy mask data from device to host
    Bit* hostMaskData = new Bit[size];
    cudaXMemcpy(hostMaskData, deviceMaskData, size * sizeof(Bit), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaXFree(deviceChoquetIntegralData);
    cudaXFree(deviceMaskData);

    // Create a shared_mask from the host data
    shared_mask result = std::make_shared<Image<Bit>>(width, height);
    result->set_data(hostMaskData);

    return result;
}

void compare_frames(shared_image host_background, std::string path, size_t nb_iter)
{
    shared_image host_image = load_png(path);

    std::cout << "Frame : " << host_image->get_width() << "x"
              << host_image->get_height() << " nb_iter: " << nb_iter << std::endl;

    static shared_bit_vector backgroundBitVector = getBitVector(host_background);
    shared_bit_vector frame = getBitVector(host_image);

    // RGB
    shared_image colorComponents = getColorImage(host_image, host_background);

    // Texture
    shared_float_vector textureComponents = getTextureComponents(frame, backgroundBitVector);

    shared_float_vector choquetIntegral = computeChoquetIntegral(textureComponents, colorComponents);

    shared_mask resultImage = getMaskResult(choquetIntegral, 0.67f);

    save_mask("dataset/results/mask_" + std::to_string(nb_iter) + ".png", resultImage);

    std::cout << "Done" << std::endl;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <path_to_dataset>" << std::endl;
        return EXIT_FAILURE;
    }

    // get all files in the directory argv[1]
    std::vector<std::string> files;
    std::string path = std::string(argv[1]);

    for (const auto& entry : std::filesystem::directory_iterator(path))
        files.push_back(entry.path());

    // sort strings in files
    std::sort(files.begin(), files.end());

    shared_image host_background = load_png(files[0]);
    for (auto it = files.begin() + 1; it != files.end(); it++)
        compare_frames(host_background, *it, it - files.begin());

    // std::string path = std::string(argv[1]);
    // shared_image host_background = load_png(path + "/1.png");
    // compare_frames(host_background, path + "/2.png", 1);

    return EXIT_SUCCESS;
}