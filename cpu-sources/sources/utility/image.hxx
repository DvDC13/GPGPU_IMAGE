
template <class T>
Image<T>::Image(int width, int height)
{
    this->width = width;
    this->height = height;
    this->data = std::vector<T>(width * height);
}

template <class T>
const T& Image<T>::get(int x, int y) const
{
    if (x < 0 || y < 0 || x >= width || y >= height)
        throw std::out_of_range("Image index out of range");
    return data[y * width + x];
}

template <class T>
T Image<T>::set(int x, int y, const T& value)
{
    if (x < 0 || y < 0 || x >= width || y >= height)
        throw std::out_of_range("Image index out of range");
    data[y * width + x] = value;
    return data[y * width + x];
}

template <class T>
T* Image<T>::operator[](int y)
{
    if (y >= height)
        throw std::out_of_range("Image index out of range");
    return &(data[y * width]);
}

#include "utility/image.h"