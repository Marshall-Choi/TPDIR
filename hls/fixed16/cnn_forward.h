#pragma once

#include "../common/cnn_dims.h"
#include "../common/ap_fixed_compat.h"

// weights.h defines: using fixed_t = ap_fixed<16, 7, ...>
#include "weights.h"

// Forward pass: Input→Q→Conv→Q→ReLU (×3) → MaxPool → Q → FC
void mnist_cnn_forward(
    const fixed_t input[CNN_IN_C][CNN_IN_H][CNN_IN_W],
    fixed_t output[CNN_NUM_CLASS]);

int mnist_cnn_predict(
    const fixed_t input[CNN_IN_C][CNN_IN_H][CNN_IN_W],
    fixed_t output[CNN_NUM_CLASS]);
