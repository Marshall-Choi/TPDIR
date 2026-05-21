#pragma once

#include "cnn_dims.h"
#include "weights.h"

// Forward pass: Input→Q→Conv→Q→ReLU (×3) → MaxPool → Q → FC  (ap_fixed<8,4>)
void mnist_cnn_forward(
    const fixed_t input[CNN_IN_C][CNN_IN_H][CNN_IN_W],
    fixed_t output[CNN_NUM_CLASS]);

int mnist_cnn_predict(
    const fixed_t input[CNN_IN_C][CNN_IN_H][CNN_IN_W],
    fixed_t output[CNN_NUM_CLASS]);
