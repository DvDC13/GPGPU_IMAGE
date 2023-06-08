#include "utility/png_utility.cuh"

namespace png_utility
{
    png_bytep* read_png_file(const std::string& filename, size_t& width,
                             size_t& height)
    {
        FILE* fp = fopen(filename.c_str(), "rb");
        if (!fp)
            return nullptr;

        png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr,
                                                 nullptr, nullptr);
        if (!png)
        {
            fclose(fp);
            return nullptr;
        }

        png_infop info = png_create_info_struct(png);
        if (!info)
        {
            png_destroy_read_struct(&png, nullptr, nullptr);
            fclose(fp);
            return nullptr;
        }

        if (setjmp(png_jmpbuf(png)))
        {
            png_destroy_read_struct(&png, &info, nullptr);
            fclose(fp);
            return nullptr;
        }

        png_init_io(png, fp);

        png_read_info(png, info);

        width = png_get_image_width(png, info);
        height = png_get_image_height(png, info);

        png_bytep* row_pointers = new png_bytep[height];

        for (size_t y = 0; y < height; y++)
            row_pointers[y] = new png_byte[png_get_rowbytes(png, info)];

        png_read_image(png, row_pointers);

        png_destroy_read_struct(&png, &info, nullptr);

        fclose(fp);

        return row_pointers;
    }

    void rows_to_float_array(png_bytep* rows, float* array, int width,
                             int height)
    {
        for (int y = 0; y < height; y++)
        {
            png_bytep row = rows[y];
            for (int x = 0; x < width; x++)
            {
                png_bytep px = &(row[x * 3]);
                array[3 * (y * width + x)] = px[0];
                array[3 * (y * width + x) + 1] = px[1];
                array[3 * (y * width + x) + 2] = px[2];
            }
        }
    }

    void float_array_to_rows(float* array, png_bytep* rows, int width,
                             int height)
    {
        for (int y = 0; y < height; y++)
        {
            png_bytep row = rows[y];
            for (int x = 0; x < width; x++)
            {
                png_bytep px = &(row[x * 3]);
                px[0] = array[3 * (y * width + x)];
                px[1] = array[3 * (y * width + x) + 1];
                px[2] = array[3 * (y * width + x) + 2];
            }
        }
    }

    void write_png_file(const std::string& filename, png_bytep* rows, int width,
                        int height)
    {
        FILE* fp = fopen(filename.c_str(), "wb");
        if (!fp)
            return;

        png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                                  nullptr, nullptr, nullptr);
        if (!png)
        {
            throw std::runtime_error("png_create_write_struct failed");
            fclose(fp);
        }

        png_infop info = png_create_info_struct(png);
        if (!info)
        {
            png_destroy_write_struct(&png, nullptr);
            fclose(fp);
            throw std::runtime_error("png_create_info_struct failed");
        }

        if (setjmp(png_jmpbuf(png)))
        {
            png_destroy_write_struct(&png, &info);
            fclose(fp);
            throw std::runtime_error("setjmp failed");
        }

        png_init_io(png, fp);

        png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB,
                     PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                     PNG_FILTER_TYPE_DEFAULT);

        png_write_info(png, info);

        png_write_image(png, rows);

        png_write_end(png, nullptr);

        png_destroy_write_struct(&png, &info);

        fclose(fp);
    }

    void bit_array_to_rows(std::vector<Bit>& bits, png_bytep* rows, int width,
                           int height)
    {
        for (int y = 0; y < height; y++)
        {
            png_bytep row = rows[y];
            for (int x = 0; x < width; x++)
            {
                png_bytep px = &(row[x * 3]);
                memset(px, bits[y * width + x] ? 255 : 0, 3);
            }
        }
    }

} // namespace png_utility
