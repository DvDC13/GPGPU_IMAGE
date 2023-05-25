#include "similarityMeasuresC.h"

float* getSimilarityMeasures(float* pixelB, float* pixelF)
{
    float* similarityMeasures = new float[3];

    float SY = std::min(pixelB[0], pixelF[0]) / std::max(pixelB[0], pixelF[0]);
    float SCr = std::min(pixelB[1], pixelF[1]) / std::max(pixelB[1], pixelF[1]);
    float SCb = std::min(pixelB[2], pixelF[2]) / std::max(pixelB[2], pixelF[2]);

    similarityMeasures[0] = SY;
    similarityMeasures[1] = SCr;
    similarityMeasures[2] = SCb;

    return similarityMeasures;
}