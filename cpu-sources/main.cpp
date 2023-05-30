#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "featuresExtractionT.h"
#include "image.h"
#include "similarityMeasuresT.h"

int main()
{
    std::string datasetPath = std::string(DATASET_DIR) + "/frames";
    std::string frame1 = datasetPath + "/1.png";
    std::string frame2 = datasetPath + "/2.png";

    shared_image image1 = load_png(frame1);
    shared_image image2 = load_png(frame2);

    return EXIT_SUCCESS;
}