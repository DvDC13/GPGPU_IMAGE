#include "texture/featuresExtractionT.h"

uint8_t isBorder(shared_image image, size_t x, size_t y)
{
    try
    {
        return getGrayscale(image->get(x, y));
    }
    catch (const std::out_of_range& e)
    {
        return 255;
    }
}

uint8_t getVector(shared_image image, int x, int y)
{
    uint8_t vector = 0;

    float gray = getGrayscale(image->get(x, y));

    vector = (vector << 1) + int(isBorder(image, x - 1, y - 1) < gray);
    vector = (vector << 1) + int(isBorder(image, x, y - 1) < gray);
    vector = (vector << 1) + int(isBorder(image, x + 1, y - 1) < gray);
    vector = (vector << 1) + int(isBorder(image, x + 1, y) < gray);
    vector = (vector << 1) + int(isBorder(image, x + 1, y + 1) < gray);
    vector = (vector << 1) + int(isBorder(image, x, y + 1) < gray);
    vector = (vector << 1) + int(isBorder(image, x - 1, y + 1) < gray);
    vector = (vector << 1) + int(isBorder(image, x - 1, y) < gray);

    return vector;
}