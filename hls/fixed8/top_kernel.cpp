#include "cnn_forward.h"

extern "C" {

void mnist_cnn(
    const fixed_t input[CNN_IN_C][CNN_IN_H][CNN_IN_W],
    fixed_t output[CNN_NUM_CLASS]) {
#pragma HLS INTERFACE mode = s_axilite port = return bundle = control
#pragma HLS INTERFACE mode = m_axi depth = 784 port = input offset = slave bundle = gmem0
#pragma HLS INTERFACE mode = m_axi depth = 10 port = output offset = slave bundle = gmem1

    mnist_cnn_forward(input, output);
}

}  // extern "C"
