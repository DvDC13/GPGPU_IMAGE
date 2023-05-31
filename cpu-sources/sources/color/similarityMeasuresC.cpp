#include "color/similarityMeasuresC.h"

float* getSimilarityMeasures(Pixel pixelB, Pixel pixelF)
{
    float* similarityMeasures = new float[3];

    if (pixelB[0] == 0.0f && pixelF[0] == 0.0f)
    {
        pixelB[0] = 1.0f;
        pixelF[0] = 1.0f;
    }

    if (pixelB[1] == 0.0f && pixelF[1] == 0.0f)
    {
        pixelB[1] = 1.0f;
        pixelF[1] = 1.0f;
    }

    if (pixelB[2] == 0.0f && pixelF[2] == 0.0f)
    {
        pixelB[2] = 1.0f;
        pixelF[2] = 1.0f;
    }

    float SY = std::min(pixelB[0], pixelF[0]) / std::max(pixelB[0], pixelF[0]);
    float SCr = std::min(pixelB[1], pixelF[1]) / std::max(pixelB[1], pixelF[1]);
    float SCb = std::min(pixelB[2], pixelF[2]) / std::max(pixelB[2], pixelF[2]);

    similarityMeasures[0] = SY;
    similarityMeasures[1] = SCr;
    similarityMeasures[2] = SCb;

    return similarityMeasures;
}