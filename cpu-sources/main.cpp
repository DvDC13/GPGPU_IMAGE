#include <fstream>
#include <iostream>

#include "featuresExtractionC.h"
#include "featuresExtractionT.h"
#include "image.h"
#include "similarityMeasuresC.h"
#include "similarityMeasuresT.h"

void m_sort(float* vec)
{

}

int main()
{
    std::string datasetPath = std::string(DATASET_DIR) + "/frames";
    std::string frame1 = datasetPath + "/1.png";
    std::string frame2 = datasetPath + "/2.png";

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
            // RGB
            float* colorRGB = getSimilarityMeasures(image1->get(x, y), image2->get(x, y));

            // Color
            Pixel yCrCb1 = RGBtoYCrCB(image1->get(x, y));
            Pixel yCrCb2 = RGBtoYCrCB(image2->get(x, y));

            image1_YCrCb->set(x, y, yCrCb1);
            image2_YCrCb->set(x, y, yCrCb2);

            // YCrCb
            float* colorComponents = getSimilarityMeasures(image1_YCrCb->get(x, y), image2_YCrCb->get(x, y));

            // Debug YCrCb This is a pixel that is in the foreground
            // if (x == 175 && y == 132)
            // {
            //     std::cout << "Pixel: " << x << " " << y << std::endl;
            //     std::cout << "image1 RGB: " << image1->get(x, y)[0] << " " << image1->get(x, y)[1] << " " << image1->get(x, y)[2] << std::endl; 
            //     std::cout << "image1 YCrCb: " << image1_YCrCb->get(x, y)[0] << " " << image1_YCrCb->get(x, y)[1] << " " << image1_YCrCb->get(x, y)[2] << std::endl;
            //     std::cout << "image2 RGB: " << image2->get(x, y)[0] << " " << image2->get(x, y)[1] << " " << image2->get(x, y)[2] << std::endl;
            //     std::cout << "image2 YCrCb: " << image2_YCrCb->get(x, y)[0] << " " << image2_YCrCb->get(x, y)[1] << " " << image2_YCrCb->get(x, y)[2] << std::endl;
            //     std::cout << "Color RGB: " << colorRGB[0] << " " << colorRGB[1] << " " << colorRGB[2] << std::endl;
            //     std::cout << "Color YCrCb: " << colorComponents[0] << " " << colorComponents[1] << " " << colorComponents[2] << std::endl;
            // }

            // Texture
            uint8_t vector1 = getVector(image1, x, y);
            uint8_t vector2 = getVector(image2, x, y);

            float textureComponent = getTextureComponent(vector1, vector2);

            // Choquet integral
            float* vector3 = new float[3];
            vector3[0] = colorComponents[0];
            vector3[1] = colorComponents[1];
            vector3[2] = textureComponent;

            // a la mano
            std::sort(vector3, vector3 + 3);

            float result = vector3[0] * weights[0] + vector3[1] * weights[1]
                + vector3[2] * weights[2];

            if (result < 0.67f)
            {
                // Foreground
                // Add a white pixel to the result image
                resultImage->set(x, y, { 255, 255, 255 });
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