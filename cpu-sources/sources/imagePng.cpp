#include "imagePng.h"

ImagePng::ImagePng()
    : m_width(0), m_height(0)
    {}

ImagePng::ImagePng(size_t width, size_t height)
    : m_width(width), m_height(height)
    {}

ImagePng::~ImagePng()
{
    if (row_pointers != nullptr)
    {
        for (size_t i = 0; i < m_height; i++)
        {
            delete[] row_pointers[i];
        }
        delete[] row_pointers;
    }
}

/*ImagePng* ImagePng::load(const char* filename)
{
    ImagePng* image = new ImagePng();

    FILE* fp = fopen(filename, "rb");
    if (!fp) return nullptr;

    auto png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) return nullptr;

    auto info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) return nullptr;

    png_init_io(png_ptr, fp);
    png_read_info(png_ptr, info_ptr);

    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, nullptr);

    auto row_pointers = png_get_rows(png_ptr, info_ptr);

    image->setWidth(png_get_image_width(png_ptr, info_ptr));
    image->setHeight(png_get_image_height(png_ptr, info_ptr));

    image->setRowPointers(row_pointers);

    free(info_ptr);
    png_destroy_read_struct(&png_ptr, nullptr, nullptr);
    fclose(fp);

    return image;
}*/

ImagePng* ImagePng::load(const char* filename)
{
    ImagePng* image = nullptr;

    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return nullptr;
    }

    png_byte header[8];
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8)) {
        std::cerr << "Invalid PNG file: " << filename << std::endl;
        fclose(fp);
        return nullptr;
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        std::cerr << "Error creating PNG read struct" << std::endl;
        fclose(fp);
        return nullptr;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        std::cerr << "Error creating PNG info struct" << std::endl;
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        fclose(fp);
        return nullptr;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        std::cerr << "Error during PNG read" << std::endl;
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        fclose(fp);
        return nullptr;
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);

    png_uint_32 width, height;
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type, nullptr, nullptr, nullptr);

    image = new ImagePng(width, height);
    image->setRowPointers(new png_bytep[height]);

    for (png_uint_32 y = 0; y < height; y++) {
        image->getRowPointers()[y] = new png_byte[png_get_rowbytes(png_ptr, info_ptr)];
    }

    png_read_image(png_ptr, image->getRowPointers());
    png_read_end(png_ptr, nullptr);

    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    fclose(fp);

    return image;
}
