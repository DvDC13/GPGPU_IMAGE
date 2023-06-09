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
    cudaXMalloc((void**)&deviceImageData, size * sizeof(Pixel));
    cudaXMalloc((void**)&deviceBitVectorData, size * sizeof(uint8_t));

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

    size_t batch_size = images.size();

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    uint8_t* backgroundBitVector = getBitVector(background);

    size_t height = images[0]->get_height();
    size_t width = images[0]->get_width();

    Pixel* backgroundData;
    cudaXMalloc((void**)&backgroundData, width * height * sizeof(Pixel));
    cudaXMemcpy(backgroundData, background->get_data().data(), width * height * sizeof(Pixel), cudaMemcpyHostToDevice);

    Pixel* imagesData;
    cudaXMalloc((void**)&imagesData, width * height * images.size() * sizeof(Pixel));
    for (size_t i = 0; i < images.size(); i++)
        cudaXMemcpy(imagesData + i * height * width, images[i]->get_data().data(), width * height * sizeof(Pixel), cudaMemcpyHostToDevice);

    Pixel* colorData;
    cudaXMalloc((void**)&colorData, width * height * images.size() * sizeof(Pixel));

    uint8_t* bitVectorData;
    cudaXMalloc((void**)&bitVectorData, width * height * images.size() * sizeof(uint8_t));

    float* textureData;
    cudaXMalloc((void**)&textureData, width * height * images.size() * sizeof(float));

    Bit* batch_masks;
    cudaXMalloc((void**)&batch_masks, width * height * images.size() * sizeof(Bit));

    for (auto it = images.begin(); it != images.end(); it += batch_size)
    {
        auto batch_end = it + batch_size;
        if (batch_end > images.end())
        {
            batch_end = images.end();
            batch_size = batch_end - it;
        }

        shared_image* batch_images = &(*it);

        dim3 blockSize(16, 16, 4);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, (batch_size + blockSize.z - 1) / blockSize.z);

        calculateSimilarityMeasures<<<gridSize, blockSize>>>(imagesData, backgroundData, colorData, it - images.begin(), batch_size, width, height);
        cudaDeviceSynchronize();

        calculateBitVector<<<gridSize, blockSize>>>(imagesData, bitVectorData, it - images.begin(), batch_size, width, height);
        cudaDeviceSynchronize();

        calculateTextureComponents<<<gridSize, blockSize>>>(bitVectorData, backgroundBitVector, textureData, it - images.begin(), batch_size, width, height);
        cudaDeviceSynchronize();

        calculateChoquetMask<<<gridSize, blockSize>>>(colorData, textureData, batch_masks, it - images.begin(), batch_size, width, height);
        cudaDeviceSynchronize();

        if (cudaPeekAtLastError())
            gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    shared_mask mask = std::make_shared<Image<Bit>>(images[0]->get_width(), images[0]->get_height());
    Bit* data = new Bit[width * height];
    for (size_t i = 0; i < images.size(); i++)
    {
        cudaXMemcpy(data, batch_masks + i * width * height, width * height * sizeof(Bit), cudaMemcpyDeviceToHost);
        mask->set_data(data);
        save_mask("dataset/results/mask_" + std::to_string(i) + ".png", mask);
    }
    delete[] data;

    float total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapsed time: " << total_time << " ms" << std::endl;
    std::cout << "FPS: " << 1000.0f / (total_time / images.size()) << std::endl;

    cudaDeviceReset();

    return EXIT_SUCCESS;
}