#pragma once

#include <png.h>
#include <iostream>
#include <stdlib.h>

class ImagePng
{
public:
    ImagePng();
    ImagePng(size_t width, size_t height);
    ~ImagePng();

    static ImagePng* load(const char* filename);

    static void save(const char* filename, ImagePng* image);

    inline size_t getWidth() const { return m_width; }
    inline size_t getHeight() const { return m_height; }
    inline png_bytep* getRowPointers() const { return row_pointers; }

    inline void setWidth(size_t width) { m_width = width; }
    inline void setHeight(size_t height) { m_height = height; }
    inline void setRowPointers(png_bytep* rowPointers) { row_pointers = rowPointers; }

private:
    png_structp png_ptr;
    png_infop info_ptr;
    png_bytep* row_pointers;
    
    size_t m_width;
    size_t m_height;
};