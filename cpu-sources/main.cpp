#include <fstream>
#include <iostream>

#include "featuresExtractionC.h"
#include "featuresExtractionT.h"
#include "image.h"
#include "similarityMeasuresC.h"
#include "similarityMeasuresT.h"

int main()
{
    std::string datasetPath = std::string(DATASET_DIR) + "/frames";
    std::string frame1 = datasetPath + "/3.png";
    std::string frame2 = datasetPath + "/4.png";

    shared_image image1 = load_png(frame1);
    shared_image image2 = load_png(frame2);

    std::cout << "Image 1: " << image1->get_width() << "x"
              << image1->get_height() << std::endl;
    std::cout << "Image 2: " << image2->get_width() << "x"
              << image2->get_height() << std::endl;

    shared_image image1_YCrCb = std::make_shared<Image<Pixel>>(
        image1->get_width(), image1->get_height());
    shared_image image2_YCrCb = std::make_shared<Image<Pixel>>(
        image2->get_width(), image2->get_height());

    shared_image resultImage = std::make_shared<Image<Pixel>>(
        image1->get_width(), image1->get_height());

    float* weights = new float[3];
    weights[0] = 0.1f;
    weights[1] = 0.3f;
    weights[2] = 0.6f;

    for (int y = 0; y < image1->get_height(); y++)
    {
        for (int x = 0; x < image1->get_width(); x++)
        {
            // Color
            // float* yCrCb1 = RGBtoYCrCB(image1->get(x, y));
            // float* yCrCb2 = RGBtoYCrCB(image2->get(x, y));

            // Pixel yCrCb1Pixel = { yCrCb1[0], yCrCb1[1], yCrCb1[2] };
            // Pixel yCrCb2Pixel = { yCrCb2[0], yCrCb2[1], yCrCb2[2] };
            // image1_YCrCb->set(x, y, yCrCb1Pixel);
            // image2_YCrCb->set(x, y, yCrCb2Pixel);
            // float* colorComponents =
            // getSimilarityMeasures(image1_YCrCb->get(x, y),
            // image2_YCrCb->get(x, y));

            float* colorComponents =
                getSimilarityMeasures(image1->get(x, y), image2->get(x, y));

            // Texture
            uint8_t vector1 = getVector(image1, x, y);
            uint8_t vector2 = getVector(image2, x, y);

            float textureComponent = getTextureComponent(vector1, vector2);

            // Choquet integral
            float* vector3 = new float[3];
            vector3[0] = colorComponents[1];
            vector3[1] = colorComponents[2];
            vector3[2] = textureComponent;

            std::sort(vector3, vector3 + 3);

            float result = vector3[0] * weights[0] + vector3[1] * weights[1]
                + vector3[2] * weights[2];

            if (result < 0.67f)
            {
                // Foreground
                // Add a white pixel to the result image
                resultImage->set(x, y, { 1, 1, 1 });
            }
            else
            {
                // Background
                // Add a black pixel to the result image
                resultImage->set(x, y, { 0, 0, 0 });
            }

            delete[] vector3;
        }
    }

    delete[] weights;

    save_png("result.png", resultImage);

    return EXIT_SUCCESS;
}