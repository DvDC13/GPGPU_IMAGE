#include "image.cuh"

shared_image load_png(const std::string filename)
{
    size_t width, height;
    png_bytep* row_pointers =
        png_utility::read_png_file(filename, width, height);

    shared_image result = std::make_shared<Image<Pixel>>(width, height);

    for (size_t y = 0; y < height; y++)
    {
        png_bytep row = row_pointers[y];
        memcpy(result->data.data()->data() + y * width * sizeof(Pixel), row,
               width * sizeof(Pixel));
    }

    for (size_t y = 0; y < height; y++)
        delete[] row_pointers[y];
    delete[] row_pointers;

    return result;
}

Pixel* load_image_batch(const std::vector<std::string> filenames)
{
    size_t width, height;
    png_bytep* row_pointers =
        png_utility::read_png_file(filename, width, height);

    Pixel* result = new Pixel[width * height * filenames.size()];

    for (size_t y = 0; y < height; y++)
    {
        png_bytep row = row_pointers[y];
        memcpy(result + y * width, row, width * sizeof(Pixel));
    }

    for (size_t y = 0; y < height; y++)
        delete[] row_pointers[y];
    delete[] row_pointers;

    for (size_t i = 1; i < filenames.size(); i++)
    {
        size_t width, height;
        png_bytep* row_pointers =
            png_utility::read_png_file(filenames[i], width, height);

        for (size_t y = 0; y < height; y++)
        {
            png_bytep row = row_pointers[y];
            memcpy(result + i * width * height + y * width, row,
                   width * sizeof(Pixel));
        }

        for (size_t y = 0; y < height; y++)
            delete[] row_pointers[y];
        delete[] row_pointers;
    }

    return result;
}

void save_png(const std::string filename, shared_image image)
{
    size_t width = image->width;
    size_t height = image->height;

    png_bytep* row_pointers = new png_bytep[height];

    for (size_t y = 0; y < height; y++)
        row_pointers[y] = new png_byte[width * 3];

    for (size_t y = 0; y < height; y++)
    {
        png_bytep row = row_pointers[y];
        memcpy(row, image->data.data()->data() + y * width * sizeof(Pixel),
               width * sizeof(Pixel));
    }

    png_utility::write_png_file(filename, row_pointers, width, height);

    for (size_t y = 0; y < height; y++)
        delete[] row_pointers[y];
    delete[] row_pointers;
}

void save_mask(const std::string filename, shared_mask mask)
{
    size_t width = mask->width;
    size_t height = mask->height;

    png_bytep* row_pointers = new png_bytep[height];

    for (size_t y = 0; y < height; y++)
        row_pointers[y] = new png_byte[width * 3];

    png_utility::bit_array_to_rows(mask->data, row_pointers, width, height);

    png_utility::write_png_file(filename, row_pointers, width, height);

    for (size_t y = 0; y < height; y++)
        delete[] row_pointers[y];
    delete[] row_pointers;
}
