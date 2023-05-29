#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "featuresExtractionT.h"
#include "image.h"
#include "similarityMeasuresT.h"

int main()
{
    // std::string datasetPath = std::string(DATASET_DIR) + "/frames";

    shared_image image = load_png("test.png");

    save_png("test2.png", image);

    return EXIT_SUCCESS;
}