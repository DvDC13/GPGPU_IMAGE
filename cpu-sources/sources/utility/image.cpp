#include "image.h"

shared_image load_png(const std::string filename)
{
    size_t width, height;
    png_bytep* row_pointers =
        png_utility::read_png_file(filename, width, height);

    shared_image result = std::make_shared<Image<Pixel>>(width, height);

    png_utility::rows_to_float_array(row_pointers, result->data.data()->data(),
                                     width, height);

    for (size_t y = 0; y < height; y++)
        delete[] row_pointers[y];
    delete[] row_pointers;

    return result;
}

void save_png(const std::string filename, shared_image image)
{
    size_t width = image->width;
    size_t height = image->height;

    png_bytep* row_pointers = new png_bytep[height];

    for (size_t y = 0; y < height; y++)
        row_pointers[y] = new png_byte[width * 3];

    png_utility::float_array_to_rows(image->data.data()->data(), row_pointers,
                                     width, height);

    png_utility::write_png_file(filename, row_pointers, width, height);

    for (size_t y = 0; y < height; y++)
        delete[] row_pointers[y];
    delete[] row_pointers;
}