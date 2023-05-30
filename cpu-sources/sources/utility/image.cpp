#include "image.h"

shared_image load_png(const char* filename)
{
    // Loading image in RGB format
    cv::Mat image = cv::imread(filename);
    if (image.empty())
        throw std::runtime_error("Failed to load the image");

    cv::Mat rgbImage;
    if (image.channels() == 3)
        cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);
    else
        throw std::runtime_error(
            "Invalid image format. Expected 3 channels (BGR)");

    // Creating shared image
    shared_image sharedImage =
        std::make_shared<Image<Pixel>>(size_t(image.cols), size_t(image.rows));

    // Copying data from cv::Mat to shared image
    for (size_t y = 0; y < sharedImage->get_height(); y++)
        for (size_t x = 0; x < sharedImage->get_width(); x++)
            sharedImage->set(x, y, rgbImage.at<Pixel>(y, x));

    return sharedImage;
}

void save_png(const std::string filename, shared_image image)
{
    // Creating cv::Mat from shared image
    cv::Mat rgbImage(image->get_height(), image->get_width(), CV_8UC3);
    for (size_t y = 0; y < image->get_height(); y++)
        for (size_t x = 0; x < image->get_width(); x++)
            rgbImage.at<Pixel>(y, x) = image->get(x, y);

    // Saving image in BGR format
    cv::Mat bgrImage;
    cv::cvtColor(rgbImage, bgrImage, cv::COLOR_RGB2BGR);
    cv::imwrite(filename, bgrImage);
}