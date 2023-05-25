#include "similarityMeasuresT.h"

float getTextureComponent(uint8_t vector1, uint8_t vector2)
{
    uint8_t vector = ~(vector1 ^ vector2);

    return __builtin_popcount(vector) / 8.0f;
}