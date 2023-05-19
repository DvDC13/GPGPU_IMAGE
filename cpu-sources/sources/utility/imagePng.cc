#include "utility/imagePng.hh"

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

void ImagePng::save(const char* filename, ImagePng* image)
{
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        std::cerr << "Error creating PNG write struct" << std::endl;
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        std::cerr << "Error creating PNG info struct" << std::endl;
        png_destroy_write_struct(&png_ptr, nullptr);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        std::cerr << "Error during PNG write" << std::endl;
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_init_io(png_ptr, fp);

    png_set_IHDR(png_ptr, info_ptr, image->getWidth(), image->getHeight(),
        8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    for (size_t y = 0; y < image->getHeight(); y++) {
        png_write_row(png_ptr, image->getRowPointers()[y]);
    }

    png_write_end(png_ptr, nullptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}
