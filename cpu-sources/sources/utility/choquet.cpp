#include "utility/choquet.h"

#include "utility/utility.h"

float compute_integral(std::array<float, 3> indicators)
{
    static std::array<float, 3> weights = { 0.1f, 0.3f, 0.6f };

    sort3<Pixel>(indicators);

    return indicators[0] * weights[0] + indicators[1] * weights[1]
        + indicators[2] * weights[2];
}