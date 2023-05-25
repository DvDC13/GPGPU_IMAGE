#include <iostream>

#include "utility/image.hh"
#include <fstream>
// #include "utility/imagePng.hh"


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

    shared_image image =
        load_png((datasetPath + "/1.png").c_str());

    std::cout << "Image width: " << image->get_width() << std::endl;
    std::cout << "Image height: " << image->get_height() << std::endl;
    std::cout << "test: " << int((*image)(1, 1)[0]) << std::endl;

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

    to_ppm(image, "test.ppm");


    return EXIT_SUCCESS;
}