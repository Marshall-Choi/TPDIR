#include "cnn_forward.h"

#include "fixed_ops.h"

#include <algorithm>

#ifndef CSIM_LOCAL
#include "hls_stream.h"
#endif

static constexpr int FW = 8;
static constexpr int FI = 4;

static fixed_t q_fixed(fixed_t x) {
#pragma HLS INLINE
    return fixed_quantize<FW, FI>(x);
}

// Synth note: PIPELINE only on innermost MAC loop + REDUCTION on acc.
// Pipelining oc/oh/ow (old code) makes HLS explode schedule size → overnight hang.
static void conv2d(
    const fixed_t in[], int in_c, int in_h, int in_w,
    const fixed_t weight[], int out_c, int k,
    const fixed_t bias[],
    fixed_t out[], int out_h, int out_w) {
#pragma HLS INLINE off

    const int out_hw = out_h * out_w;

    for (int oc = 0; oc < out_c; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                fixed_t acc = bias[oc];
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int kh = 0; kh < k; ++kh) {
                        for (int kw = 0; kw < k; ++kw) {
#pragma HLS PIPELINE II = 1
#pragma HLS REDUCTION variable = acc
                            const int ih = oh + kh;
                            const int iw = ow + kw;
                            const int i_idx = ic * (in_h * in_w) + ih * in_w + iw;
                            const int w_idx = oc * (in_c * k * k) + ic * (k * k) + kh * k + kw;
                            acc += in[i_idx] * weight[w_idx];
                        }
                    }
                }
                out[oc * out_hw + oh * out_w + ow] = acc;
            }
        }
    }
}

static void apply_relu(fixed_t buf[], int n) {
    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
        buf[i] = relu(buf[i]);
    }
}

static void apply_quant(fixed_t buf[], int n) {
    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
        buf[i] = q_fixed(buf[i]);
    }
}

static void maxpool2x2(
    const fixed_t in[], int c, int in_h, int in_w,
    fixed_t out[], int out_h, int out_w) {
    const int in_hw = in_h * in_w;
    const int out_hw = out_h * out_w;

    for (int ch = 0; ch < c; ++ch) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
#pragma HLS PIPELINE II = 1
                fixed_t best = in[ch * in_hw + (oh * 2) * in_w + (ow * 2)];
                for (int kh = 0; kh < 2; ++kh) {
                    for (int kw = 0; kw < 2; ++kw) {
                        const fixed_t v = in[ch * in_hw + (oh * 2 + kh) * in_w + (ow * 2 + kw)];
                        if (v > best) best = v;
                    }
                }
                out[ch * out_hw + oh * out_w + ow] = best;
            }
        }
    }
}

static void linear(
    const fixed_t in[], int in_features,
    const fixed_t weight[], const fixed_t bias[],
    fixed_t out[], int out_features) {
#pragma HLS INLINE off

    for (int o = 0; o < out_features; ++o) {
        fixed_t acc = bias[o];
        for (int i = 0; i < in_features; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS REDUCTION variable = acc
            acc += in[i] * weight[o * in_features + i];
        }
        out[o] = acc;
    }
}

void mnist_cnn_forward(
    const fixed_t input[CNN_IN_C][CNN_IN_H][CNN_IN_W],
    fixed_t output[CNN_NUM_CLASS]) {

    static fixed_t buf_a[CNN_C1_OUT * CNN_H1 * CNN_W1];
    static fixed_t buf_b[CNN_C2_OUT * CNN_H2 * CNN_W2];
    static fixed_t buf_c[CNN_C3_OUT * CNN_H3 * CNN_W3];
    static fixed_t buf_p[CNN_C3_OUT * CNN_POOL_H * CNN_POOL_W];
    static fixed_t flat[CNN_FLAT];
    static fixed_t in_q[CNN_IN_C * CNN_IN_H * CNN_IN_W];

#pragma HLS BIND_STORAGE variable = buf_a type = RAM_2P impl = BRAM
#pragma HLS BIND_STORAGE variable = buf_b type = RAM_2P impl = BRAM
#pragma HLS BIND_STORAGE variable = buf_c type = RAM_2P impl = BRAM
#pragma HLS BIND_STORAGE variable = buf_p type = RAM_2P impl = BRAM
#pragma HLS BIND_STORAGE variable = flat type = RAM_2P impl = BRAM
#pragma HLS BIND_STORAGE variable = in_q type = RAM_2P impl = BRAM

    for (int h = 0; h < CNN_IN_H; ++h) {
        for (int w = 0; w < CNN_IN_W; ++w) {
#pragma HLS PIPELINE II = 1
            in_q[h * CNN_IN_W + w] = q_fixed(input[0][h][w]);
        }
    }

    conv2d(in_q, CNN_IN_C, CNN_IN_H, CNN_IN_W,
           &CONV1_WEIGHT[0][0][0][0], CNN_C1_OUT, CNN_K,
           CONV1_BIAS, buf_a, CNN_H1, CNN_W1);
    apply_quant(buf_a, CNN_C1_OUT * CNN_H1 * CNN_W1);
    apply_relu(buf_a, CNN_C1_OUT * CNN_H1 * CNN_W1);

    conv2d(buf_a, CNN_C1_OUT, CNN_H1, CNN_W1,
           &CONV2_WEIGHT[0][0][0][0], CNN_C2_OUT, CNN_K,
           CONV2_BIAS, buf_b, CNN_H2, CNN_W2);
    apply_quant(buf_b, CNN_C2_OUT * CNN_H2 * CNN_W2);
    apply_relu(buf_b, CNN_C2_OUT * CNN_H2 * CNN_W2);

    conv2d(buf_b, CNN_C2_OUT, CNN_H2, CNN_W2,
           &CONV3_WEIGHT[0][0][0][0], CNN_C3_OUT, CNN_K,
           CONV3_BIAS, buf_c, CNN_H3, CNN_W3);
    apply_quant(buf_c, CNN_C3_OUT * CNN_H3 * CNN_W3);
    apply_relu(buf_c, CNN_C3_OUT * CNN_H3 * CNN_W3);

    maxpool2x2(buf_c, CNN_C3_OUT, CNN_H3, CNN_W3, buf_p, CNN_POOL_H, CNN_POOL_W);
    apply_quant(buf_p, CNN_C3_OUT * CNN_POOL_H * CNN_POOL_W);

    for (int i = 0; i < CNN_FLAT; ++i) {
#pragma HLS PIPELINE II = 1
        flat[i] = buf_p[i];
    }

    linear(flat, CNN_FLAT, &FC_WEIGHT[0][0], FC_BIAS, output, CNN_NUM_CLASS);
}

int mnist_cnn_predict(
    const fixed_t input[CNN_IN_C][CNN_IN_H][CNN_IN_W],
    fixed_t output[CNN_NUM_CLASS]) {
    mnist_cnn_forward(input, output);
    int best = 0;
    for (int i = 1; i < CNN_NUM_CLASS; ++i) {
        if (float(output[i]) > float(output[best])) best = i;
    }
    return best;
}
