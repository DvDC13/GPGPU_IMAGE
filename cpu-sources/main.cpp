#include <iostream>

#include "image.h"
#include <fstream>
// #include "utility/imagePng.hh"

#include "featuresExtractionT.h"
#include "similarityMeasuresT.h"

void to_ppm(shared_image image, std::string name)
{
    std::ofstream file(name);
    file << "P3" << std::endl;
    file << image->get_width() << " " << image->get_height() << std::endl;
    file << "255" << std::endl;

    for (size_t y = 0; y < image->get_height(); y++)
    {
        for (size_t x = 0; x < image->get_width(); x++)
        {
            file << int((*image)(x, y)[0]) << " "
                 << int((*image)(x, y)[1]) << " "
                 << int((*image)(x, y)[2]) << " ";
        }
        file << std::endl;
    }
}

int main()
{
    std::string datasetPath = std::string(DATASET_DIR) + "/frames";

    shared_image imageB =
        load_png((datasetPath + "/1.png").c_str());

    shared_image imageF =
        load_png((datasetPath + "/2.png").c_str());

    uint8_t vectorB = getVector(imageB, 0, 0);
    std::cout << "Vector: " << int(vectorB) << std::endl;
    uint8_t vectorF = getVector(imageF, 0, 0);
    std::cout << "Vector: " << int(vectorF) << std::endl;

    float tc = getTextureComponent(vectorB, vectorF);
    std::cout << "Texture component: " << tc << std::endl;

    // Print all pixels in the image
    // for (size_t y = 0; y < image->get_height(); y++)
    // {
    //     for (size_t x = 0; x < image->get_width(); x++)
    //     {
    //         std::cout << "(" << int((*image)(x, y)[0]) << ", "
    //                   << int((*image)(x, y)[1]) << ", "
    //                   << int((*image)(x, y)[2]) << ") ";
    //     }
    //     std::cout << std::endl;
    // }

    // to_ppm(image, "test.ppm");


    return EXIT_SUCCESS;
}