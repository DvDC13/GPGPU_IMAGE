
template <class T>
Image<T>::Image(size_t width, size_t height)
{
    this->width = width;
    this->height = height;
    this->data = std::vector<T>(width * height);
}

template <class T>
const T& Image<T>::operator()(size_t x, size_t y) const
{
    return data[y * width + x];
}

template <class T>
T& Image<T>::operator()(size_t x, size_t y)
{
    return data[y * width + x];
}