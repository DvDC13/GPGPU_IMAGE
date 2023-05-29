
template <class T>
Image<T>::Image(size_t width, size_t height)
{
    this->width = width;
    this->height = height;
    this->data = std::vector<T>(width * height);
}

template <class T>
const T& Image<T>::get(size_t x, size_t y) const
{
    if (x >= width || y >= height)
        throw std::out_of_range("Image index out of range");
    return data[y * width + x];
}

template <class T>
T& Image<T>::set(size_t x, size_t y, const T& value)
{
    if (x >= width || y >= height)
        throw std::out_of_range("Image index out of range");
    return data[y * width + x] = value;
}

template <class T>
T* Image<T>::operator[](size_t y)
{
    if (y >= height)
        throw std::out_of_range("Image index out of range");
    return &data[y * width];
}