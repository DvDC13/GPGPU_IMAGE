#include <iostream>

#include "imagePng.h"
#include "featuresExtraction.h"
#include "similarityMeasures.h"

int main()
{
    std::string datasetPath = std::string(DATASET_DIR) + "/frames";

    ImagePng* imageB = ImagePng::load((datasetPath + "/1.png").c_str());
    ImagePng* imageF = ImagePng::load((datasetPath + "/2.png").c_str());

    for (size_t y = 0; y < imageB->getHeight(); y++) {
        for (size_t x = 0; x < imageB->getWidth(); x++) {
            png_byte* pixelB = &(imageB->getRowPointers()[y][x * 3]);
            png_byte* pixelF = &(imageF->getRowPointers()[y][x * 3]);

            float* yCrCb_backg = RGBtoYCrCB((unsigned char*)pixelB);
            float* yCrCb_frame = RGBtoYCrCB((unsigned char*)pixelF);

            float* RGB_F = new float[3];
            RGB_F[0] = (float)pixelF[0];
            RGB_F[1] = (float)pixelF[1];
            RGB_F[2] = (float)pixelF[2];

            float* RGB_B = new float[3];
            RGB_B[0] = (float)pixelB[0];
            RGB_B[1] = (float)pixelB[1];
            RGB_B[2] = (float)pixelB[2];

            float* s = getSimilarityMeasures(yCrCb_backg, yCrCb_frame);
            std::cout << s[0] << " " << s[1] << " " << s[2] << std::endl;

            delete[] s;
        }
        std::cout << std::endl;
    }

    ImagePng::save("test.png", imageB);

    delete imageB;

    return EXIT_SUCCESS;
}