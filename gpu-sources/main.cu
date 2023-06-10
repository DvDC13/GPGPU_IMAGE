#include <algorithm>
#include <filesystem>

#include "wrapper_gpu.cuh"
#include "similarityMeasuresC.cuh"
#include "featuresExtractionT.cuh"
#include "similarityMeasuresT.cuh"
#include "choquet.cuh"

#include "image.cuh"

cudaStream_t stream1;
cudaStream_t stream2;

uint8_t* getBitVector(shared_image image)
{
    int width = image->get_width();
    int height = image->get_height();
    int size = width * height;

    // Allocate device memory
    Pixel* deviceImageData;
    uint8_t* deviceBitVectorData;
    cudaXCalloc((void**)&deviceImageData, size * sizeof(Pixel));
    cudaXCalloc((void**)&deviceBitVectorData, size * sizeof(uint8_t));

    // Copy image data from host to device
    cudaXMemcpy(deviceImageData, image->get_data().data(), size * sizeof(Pixel), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    calculateBitVectorBackground<<<gridSize, blockSize>>>(deviceImageData, deviceBitVectorData, width, height);
    cudaDeviceSynchronize();

    return deviceBitVectorData;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <path_to_dataset>" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::string> files;
    std::string path = std::string(argv[1]);

    std::vector<shared_image> images;

    for (const auto& entry : std::filesystem::directory_iterator(path))
        files.push_back(entry.path());

    std::sort(files.begin(), files.end());

    files.reserve(files.size() - 1);

    for (auto it = files.begin() + 1; it != files.end(); it++)
        images.push_back(load_png(*it));

    shared_image background = load_png(files[0]);

    size_t height = images[0]->get_height();
    size_t width = images[0]->get_width();

    size_t memory_usage = width * height * ((sizeof(Pixel)) + sizeof(uint8_t) + sizeof(float) + sizeof(Bit) + sizeof(std::array<float, 2>));
    size_t memory_usage_bg = width * height * (sizeof(uint8_t) + sizeof(Pixel));

    size_t maximum_global_memory = 0;
    cudaMemGetInfo(&maximum_global_memory, nullptr);
    size_t max_batch_size = std::floor((maximum_global_memory - memory_usage_bg) / memory_usage);
    size_t batch_size = std::min(max_batch_size, images.size());

    std::cout << "Height: " << height << std::endl;
    std::cout << "Width: " << width << std::endl;
    std::cout << "Memory usage: " << memory_usage << std::endl;
    std::cout << "Memory usage background: " << memory_usage_bg << std::endl;
    std::cout << "Maximum global memory: " << maximum_global_memory << std::endl;
    std::cout << "Maximum batch size: " << max_batch_size << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    cudaStream_t stream1;
    cudaStream_t stream2;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    uint8_t* backgroundBitVector = getBitVector(background);

    Pixel* backgroundData;
    cudaXCalloc((void**)&backgroundData, width * height * sizeof(Pixel));
    cudaXMemcpy(backgroundData, background->get_data().data(), width * height * sizeof(Pixel), cudaMemcpyHostToDevice);

    Pixel* imagesData;
    cudaXCalloc((void**)&imagesData, width * height * batch_size * sizeof(Pixel));

    std::array<float, 2>* colorData;
    cudaXCalloc((void**)&colorData, width * height * batch_size * sizeof(std::array<float, 2>));

    uint8_t* bitVectorData;
    cudaXCalloc((void**)&bitVectorData, width * height * batch_size * sizeof(uint8_t));

    float* textureData;
    cudaXCalloc((void**)&textureData, width * height * batch_size * sizeof(float));

    Bit* batch_masks;
    cudaXCalloc((void**)&batch_masks, width * height * batch_size * sizeof(Bit));

    Bit* data_to_save;
    cudaXMallocHost((void**)&data_to_save, width * height * images.size() * sizeof(Bit));

    for (auto it = images.begin(); it != images.end(); it += batch_size)
    {
        auto batch_end = it + batch_size;
        if (batch_end > images.end())
        {
            batch_end = images.end();
            batch_size = batch_end - it;
        }

        for (size_t i = 0; i < batch_size; i++)
            cudaXMemcpy(imagesData + i * width * height, images[it - images.begin() + i]->get_data().data(), width * height * sizeof(Pixel), cudaMemcpyHostToDevice);

        shared_image* batch_images = &(*it);

        dim3 blockSize(16, 16, 4);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, (batch_size + blockSize.z - 1) / blockSize.z);

        calculateSimilarityMeasures<<<gridSize, blockSize, 0, stream1>>>(imagesData, backgroundData, colorData, batch_size, width, height);
        calculateBitVector<<<gridSize, blockSize, 0, stream2>>>(imagesData, bitVectorData, batch_size, width, height);
        calculateTextureComponents<<<gridSize, blockSize, 0, stream2>>>(bitVectorData, backgroundBitVector, textureData, batch_size, width, height);
        cudaDeviceSynchronize();

        calculateChoquetMask<<<gridSize, blockSize>>>(colorData, textureData, batch_masks, batch_size, width, height);
        cudaDeviceSynchronize();

        cudaXMemcpy(data_to_save + (it - images.begin()) * width * height, batch_masks, width * height * batch_size * sizeof(Bit), cudaMemcpyDeviceToHost);

        if (cudaPeekAtLastError())
            gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    shared_mask mask = std::make_shared<Image<Bit>>(width, height);
    for (size_t i = 0; i < images.size(); i++)
    {
        mask->set_data(data_to_save + i * width * height);
        char nb[6];
        snprintf(nb, 6, "%05lu", i);
        save_mask("dataset/results/mask_" + std::string(nb) + ".png", mask);
    }

    float total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapsed time: " << total_time << " ms" << std::endl;
    float fps = 1000.0f / (total_time / images.size());
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "PPS: " << fps * width * height << std::endl; 

    cudaXFree(backgroundData);
    cudaXFree(imagesData);
    cudaXFree(colorData);
    cudaXFree(bitVectorData);
    cudaXFree(textureData);
    cudaXFree(batch_masks);
    cudaXFreeHost(data_to_save);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}
