#pragma once

#include <cmath>

#include "ap_fixed_compat.h"

// ap_fixed<W,I> quantize — matches quant_models.make_fixed_round / PyTorch QAT
template <int W, int I, typename T>
T fixed_quantize(T x) {
    const int F = W - I;
    const float scale = float(1 << F);
    const float vmin = float(-(1 << (I - 1)));
    const float vmax = float((1 << (I - 1)) - (1.0f / scale));

    float xf = float(x);
    if (xf < vmin) xf = vmin;
    if (xf > vmax) xf = vmax;
    xf = std::round(xf * scale) / scale;
    return T(xf);
}

template <typename T>
T relu(T x) {
    return x > T(0) ? x : T(0);
}
