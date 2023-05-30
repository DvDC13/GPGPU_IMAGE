#pragma once

#include <array>
#include <memory>
#include <opencv2/opencv.hpp>
#include <png.h>
#include <stdexcept>
#include <string.h>
#include <string>
#include <vector>

template <class T>
class Image;

using Pixel = cv::Vec3b;
using Bit = bool;
using shared_image = std::shared_ptr<Image<Pixel>>;

shared_image load_png(const std::string filename);
void save_png(const std::string filename, shared_image image);

template <class T>
class Image
{
public:
    Image(int width, int height);

    friend shared_image load_png(const char* filename);
    friend void save_png(const std::string filename, shared_image image);

    // Get a pixel from the image
    const T& get(int x, int y) const;

    // Set a pixel in the image
    T& set(int x, int y, const T& value);

    // Operator to have a more intuitive way to access the image
    // WARNING: This operator is not bounds checked
    T* operator[](int y);

    inline int get_width() const
    {
        return width;
    }

    inline int get_height() const
    {
        return height;
    }

private:
    int width;
    int height;
    std::vector<T> data;
};

#include "utility/image.hxx"