#include "color/featuresExtractionC.h"

Pixel RGBtoYCrCB(Pixel pixel)
{
    Pixel yCrCb = std::array<float, 3>();

    float Y = 0.25 * pixel[0] + 0.504 * pixel[1] + 0.098 * pixel[2] + 16;
    float Cr = 0.439 * pixel[0] - 0.368 * pixel[1] - 0.071 * pixel[2] + 128;
    float Cb = -0.148 * pixel[0] - 0.291 * pixel[1] + 0.439 * pixel[2] + 128;

    yCrCb[0] = Y;
    yCrCb[1] = Cr;
    yCrCb[2] = Cb;

    return yCrCb;
}