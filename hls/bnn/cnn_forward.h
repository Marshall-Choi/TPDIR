#pragma once

#include "../common/cnn_dims.h"

#include "weights.h"

// BNN: sign → bin_conv → BN(float) → sign → … → pool → FC(float)
void mnist_bnn_forward(
    const float input[CNN_IN_C][CNN_IN_H][CNN_IN_W],
    float output[CNN_NUM_CLASS]);

int mnist_bnn_predict(
    const float input[CNN_IN_C][CNN_IN_H][CNN_IN_W],
    float output[CNN_NUM_CLASS]);
