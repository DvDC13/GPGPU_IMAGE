#include "texture/featuresExtractionT.h"

uint8_t isBorder(shared_image image, int x, int y)
{
    if (x < 0 || y < 0 || x > image->get_width() - 1 || y > image->get_height() - 1)
        return 255;
    
    return getGrayscale(image->get(x, y));
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