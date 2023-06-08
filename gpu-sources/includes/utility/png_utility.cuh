#pragma once

#include <memory>
#include <png.h>
#include <string>
#include <vector>

#include "utility/image.cuh"

namespace png_utility
{
    using Bit = bool;

    png_bytep* read_png_file(const std::string& filename, size_t& width,
                             size_t& height);

    void rows_to_float_array(png_bytep* rows, float* array, int width,
                             int height);

    void write_png_file(const std::string& filename, png_bytep* rows, int width,
                        int height);

    void float_array_to_rows(float* array, png_bytep* rows, int width,
                             int height);

    void bit_array_to_rows(std::vector<Bit>& bits, png_bytep* rows, int width,
                           int height);

} // namespace png_utility
