#include "color/similarityMeasuresC.h"

uint8_t min(char a, char b)
{
    uint8_t res = a <= b ? a : b;
    return res;
}

uint8_t max(char a, char b)
{
    uint8_t res = a >= b ? a : b;
    return res;
}

float* getSimilarityMeasures(Pixel pixelB, Pixel pixelF)
{
    float* similarityMeasures = new float[3];

    if (pixelB[0] == 0 && pixelF[0] == 0)
    {
        pixelB[0] = 1;
        pixelF[0] = 1;
    }

    if (pixelB[1] == 0 && pixelF[1] == 0)
    {
        pixelB[1] = 1;
        pixelF[1] = 1;
    }

    if (pixelB[2] == 0 && pixelF[2] == 0)
    {
        pixelB[2] = 1;
        pixelF[2] = 1;
    }

    float SY = (float)min(pixelB[0], pixelF[0]) / (float)max(pixelB[0], pixelF[0]);
    float SCr = (float)min(pixelB[1], pixelF[1]) / (float)max(pixelB[1], pixelF[1]);
    float SCb = (float)min(pixelB[2], pixelF[2]) / (float)max(pixelB[2], pixelF[2]);

    similarityMeasures[0] = SY;
    similarityMeasures[1] = SCr;
    similarityMeasures[2] = SCb;

    return similarityMeasures;
}