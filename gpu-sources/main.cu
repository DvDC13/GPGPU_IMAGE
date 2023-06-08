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
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    calculateBitVector<<<gridSize, blockSize>>>(deviceImageData, deviceBitVectorData, width, height);
    //cudaDeviceSynchronize();

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
    dim3 blockSize(32, 32);
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
    dim3 blockSize(32, 32);
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
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    calculateChoquetIntegral<<<gridSize, blockSize>>>(deviceColorComponentsData, deviceTextureComponentsData, deviceChoquetIntegralData, width, height);
    cudaDeviceSynchronize();

    // Copy choquet integral data from device to host
    std::vector<float> hostChoquetIntegralData(size);
    cudaXMemcpy(hostChoquetIntegralData.data(), deviceChoquetIntegralData, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaXFree(deviceTextureComponentsData);
    cudaXFree(deviceColorComponentsData);
    cudaXFree(deviceChoquetIntegralData);

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
    dim3 blockSize(32, 32);
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

std::vector<shared_mask> compare_batches(shared_image host_background, std::vector<shared_image>& host_images, shared_bit_vector backgroundBitVector, size_t batch_size)
{
    std::vector<shared_bit_vector> batch_frames(batch_size);
    std::vector<shared_image> batch_color_components(batch_size);
    std::vector<shared_float_vector> batch_texture_components(batch_size);
    std::vector<shared_float_vector> batch_choquet_integrals(batch_size);
    std::vector<shared_mask> batch_masks(batch_size);

    #pragma omp parallel for
    for (size_t i = 0; i < batch_size; i++)
    {
        size_t frame_idx = i;
        shared_image frame = host_images[frame_idx];
        
        batch_frames[i] = getBitVector(frame);
        batch_color_components[i] = getColorImage(frame, host_background);
        batch_texture_components[i] = getTextureComponents(batch_frames[i], backgroundBitVector);
        batch_choquet_integrals[i] = computeChoquetIntegral(batch_texture_components[i], batch_color_components[i]);
        batch_masks[i] = getMaskResult(batch_choquet_integrals[i], 0.67f);
    }

    return batch_masks;
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

    size_t batch_size = 10;

    std::vector<std::vector<shared_mask>> masks;

    shared_bit_vector backgroundBitVector = getBitVector(background);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (auto it = images.begin() + 1; it != images.end(); it += batch_size)
    {
        auto batch_end = it + batch_size;
        if (batch_end > images.end())
        {
            batch_end = images.end();
            batch_size = batch_end - it;
        }

        std::vector<shared_image> batch_images(it, batch_end);
        std::vector<shared_mask> batch_masks = compare_batches(background, batch_images, backgroundBitVector, batch_size);

        masks.push_back(batch_masks);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    for (size_t i = 0; i < masks.size(); i++)
    {
        for (size_t j = 0; j < masks[i].size(); j++)
        {
            save_mask("dataset/results/mask_" + std::to_string(i * masks[i].size() + j) + ".png", masks[i][j]);
        }
    }

    float total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapsed time: " << total_time << " ms" << std::endl;
    std::cout << "FPS: " << 1000.0f / (total_time / images.size()) << std::endl;

    cudaDeviceReset();

    return EXIT_SUCCESS;
}