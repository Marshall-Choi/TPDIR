#pragma once

// MNIST CNN topology (matches quant_models.py / EEE429 spec)
#define CNN_IN_H 28
#define CNN_IN_W 28
#define CNN_IN_C 1

#define CNN_C1_OUT 16
#define CNN_C2_OUT 32
#define CNN_C3_OUT 32

#define CNN_K 3

#define CNN_H1 26
#define CNN_W1 26
#define CNN_H2 24
#define CNN_W2 24
#define CNN_H3 22
#define CNN_W3 22

#define CNN_POOL_H 11
#define CNN_POOL_W 11
#define CNN_FLAT 3872

#define CNN_NUM_CLASS 10
