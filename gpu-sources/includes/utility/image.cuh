#pragma once

#include <cuda_runtime.h>
#include <array>
#include <iostream>
#include <memory>
#include <png.h>
#include <stdexcept>
#include <string.h>
#include <string>
#include <vector>

#include "utility/png_utility.cuh"

template <class T>
class Image;

using Pixel = std::array<float, 3>;
using Bit = bool;
using shared_image = std::shared_ptr<Image<Pixel>>;
using shared_mask = std::shared_ptr<Image<Bit>>;
using shared_bit_vector = std::shared_ptr<Image<uint8_t>>;
using shared_float_vector = std::shared_ptr<Image<float>>;

shared_image load_png(const std::string filename);
void save_png(const std::string filename, shared_image image);

template <class T>
class Image
{
public:
    Image(int width, int height);

    friend shared_image load_png(const std::string filename);
    friend void save_png(const std::string filename, shared_image image);
    friend void save_mask(const std::string filename, shared_mask image);

    // Get a pixel from the image
    const T& get(int x, int y) const;

    // Set a pixel in the image
    T set(int x, int y, const T& value);

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

    inline std::vector<T>& get_data()
    {
        return data;
    }

    inline void set_data(std::vector<T> data)
    {
        this->data = data;
    }
    
    inline void set_data(T* data)
    {
        this->data = std::vector<T>(data, data + width * height);
    }

private:
    int width;
    int height;
    std::vector<T> data;
};

#include "utility/image.cuhxx"