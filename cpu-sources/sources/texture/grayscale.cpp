#include "grayscale.h"

float getGrayscale(Pixel pixel)
{
    float grayscale = 0.0f;

    grayscale = 0.3 * pixel[0] + 0.59 * pixel[1] + 0.11 * pixel[2];

    return grayscale;
}