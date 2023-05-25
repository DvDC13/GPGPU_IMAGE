#include "utility/image.hh"

shared_image load_png(const char* filename)
{
    FILE* fp = fopen(filename, "rb");
    if (!fp)
        throw std::runtime_error("Error opening file");

    png_byte header[8];
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8))
    {
        fclose(fp);
        return nullptr;
        throw std::runtime_error("Invalid PNG file");
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr,
                                                 nullptr, nullptr);
    if (!png_ptr)
    {
        fclose(fp);
        throw std::runtime_error("Error creating PNG read struct");
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        fclose(fp);
        throw std::runtime_error("Error creating PNG info struct");
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        fclose(fp);
        throw std::runtime_error("Error during PNG read");
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);

    png_uint_32 width, height;
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type,
                 nullptr, nullptr, nullptr);

    shared_image image = std::make_shared<Image<Pixel>>(width, height);
    png_bytep* row_pointers = nullptr;

    row_pointers = new png_bytep[height];

    for (png_uint_32 y = 0; y < height; y++)
        row_pointers[y] = new png_byte[png_get_rowbytes(png_ptr, info_ptr)];

    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, nullptr);

    for (png_uint_32 y = 0; y < height; y++)
        memccpy((*image).data.data() + y * width, row_pointers[y], 1,
                width * sizeof(Pixel));

    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    fclose(fp);

    return image;
}

void save_png(const std::string filename, shared_image image)
{
    // // Open the PNG image file
    // FILE* file = fopen(filename.c_str(), "wb");
    // if (!file)
    // {
    //     // Handle error: Unable to open the file
    //     throw std::runtime_error("Unable to open the file");
    // }

    // // Create PNG write structures
    // png_structp png =
    //     png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    // if (!png)
    // {
    //     // Handle error: Unable to create PNG write struct
    //     fclose(file);
    //     // Clean up any allocated resources
    //     throw std::runtime_error("Unable to create PNG write struct");
    // }

    // png_infop info = png_create_info_struct(png);
    // if (!info)
    // {
    //     // Handle error: Unable to create PNG info struct
    //     png_destroy_write_struct(&png, NULL);
    //     fclose(file);
    //     // Clean up any allocated resources
    //     throw std::runtime_error("Unable to create PNG info struct");
    // }

    // // Set error handling
    // if (setjmp(png_jmpbuf(png)))
    // {
    //     // Handle error: PNG error occurred during initialization
    //     png_destroy_write_struct(&png, &info);
    //     fclose(file);
    //     // Clean up any allocated resources
    //     throw std::runtime_error("PNG error occurred during initialization");
    // }

    // // Initialize PNG IO
    // png_init_io(png, file);

    // // Write PNG info
    // png_set_IHDR(png, info, image->width, image->height, 8,
    // PNG_COLOR_TYPE_RGBA,
    //              PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
    //              PNG_FILTER_TYPE_DEFAULT);
    // png_write_info(png, info);

    // // Write image data
    // png_bytepp rowPointers = new png_bytep[image->height];
    // for (png_uint_32 y = 0; y < image->height; ++y)
    //     rowPointers[y] = image->data[y * image->width * 4];
    // png_write_image(png, rowPointers);
    // png_write_end(png, NULL);

    // // Clean up resources
    // png_destroy_write_struct(&png, &info);
    // fclose(file);
    // delete[] rowPointers;
}