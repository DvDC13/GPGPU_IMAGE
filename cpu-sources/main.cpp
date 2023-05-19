#include <iostream>

#include "imagePng.h"

int main()
{
    std::string datasetPath = std::string(DATASET_DIR) + "/frames";

    ImagePng* image = ImagePng::load((datasetPath + "/1.png").c_str());

    std::cout << "Width: " << image->getWidth() << std::endl;
    std::cout << "Height: " << image->getHeight() << std::endl;

    for (size_t y = 0; y < image->getHeight(); y++) {
        for (size_t x = 0; x < image->getWidth(); x++) {
            png_byte* pixel = &(image->getRowPointers()[y][x * 3]);
            std::cout << "R: " << (int)pixel[0] << " ";
            std::cout << "G: " << (int)pixel[1] << " ";
            std::cout << "B: " << (int)pixel[2] << " ";
        }
        std::cout << std::endl;
    }

    ImagePng::save("test.png", image);

    delete image;

    return EXIT_SUCCESS;
}