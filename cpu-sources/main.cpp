#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "featuresExtractionC.h"
#include "featuresExtractionT.h"
#include "image.h"
#include "similarityMeasuresC.h"
#include "similarityMeasuresT.h"
#include "utility/choquet.h"

void log(std::string status, std::string message)
{
    std::cerr << "[" << status << "] " << message << '\n';
}

shared_bit_vector getBitVector(shared_image image)
{
    shared_bit_vector result = std::make_shared<Image<uint8_t>>(
        image->get_width(), image->get_height());

    for (int y = 0; y < image->get_height(); y++)
        for (int x = 0; x < image->get_width(); x++)
            result->set(x, y, getVector(image, x, y));

    return result;
}

void compare_frames(shared_image background, shared_image image2,
                    size_t nb_iter)
{
    std::cout << "frame: " << image2->get_width() << "x" << image2->get_height()
              << " nb_iter: " << nb_iter << std::endl;

    shared_mask resultImage = std::make_shared<Image<Bit>>(
        background->get_width(), background->get_height());

    static shared_bit_vector backgroundBitVector = getBitVector(background);
    shared_bit_vector frame = getBitVector(image2);

    for (int y = 0; y < background->get_height(); y++)
    {
        for (int x = 0; x < background->get_width(); x++)
        {
            // RGB
            std::array<float, 3> colorRGB =
                getSimilarityMeasures(background->get(x, y), image2->get(x, y));

            // Texture
            uint8_t vector1 = backgroundBitVector->get(x, y);
            uint8_t vector2 = frame->get(x, y);
            float textureComponent = getTextureComponent(vector1, vector2);

            float result = compute_integral(
                { colorRGB[0], colorRGB[1], textureComponent });

            const bool isForeground = (result < 0.67) ? true : false;
            resultImage->set(x, y, isForeground);
        }
    }

    save_mask("dataset/results/mask_" + std::to_string(nb_iter) + ".png",
              resultImage);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <path to dataset>" << std::endl;
        return EXIT_FAILURE;
    }

    // get all files in the directory argv[1]
    std::vector<std::string> files;
    std::string path = std::string(argv[1]);

    std::vector<shared_image> images;

    for (const auto& entry : std::filesystem::directory_iterator(path))
        files.push_back(entry.path());

    // sort strings in files
    std::sort(files.begin(), files.end());

    files.reserve(files.size() - 1);

    for (auto it = files.begin() + 1; it != files.end(); it++)
        images.push_back(load_png(*it));

    shared_image background = load_png(files[0]);
    for (auto it = images.begin() + 1; it != images.end(); it++)
        compare_frames(background, *it, it - images.begin());

    return EXIT_SUCCESS;
}