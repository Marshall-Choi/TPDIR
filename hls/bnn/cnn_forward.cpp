#include "cnn_forward.h"

#include <cmath>

#ifndef CSIM_LOCAL
#include "hls_stream.h"
#endif

static constexpr float BN_EPS = 1e-5f;

static float bin_sign(float x) {
#pragma HLS INLINE
    return x >= 0.f ? 1.f : -1.f;
}

static void apply_bin_sign(float buf[], int n) {
    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II = 1
        buf[i] = bin_sign(buf[i]);
    }
}

// Binary conv: input ±1 float, weight int8 ±1, no bias
static void bin_conv2d(
    const float in[], int in_c, int in_h, int in_w,
    const int8_t weight[], int out_c, int k,
    float out[], int out_h, int out_w) {
    const int out_hw = out_h * out_w;

    for (int oc = 0; oc < out_c; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
#pragma HLS PIPELINE II = 1
                float acc = 0.f;
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int kh = 0; kh < k; ++kh) {
                        for (int kw = 0; kw < k; ++kw) {
                            const int ih = oh + kh;
                            const int iw = ow + kw;
                            const int i_idx = ic * (in_h * in_w) + ih * in_w + iw;
                            const int w_idx = oc * (in_c * k * k) + ic * (k * k) + kh * k + kw;
                            acc += in[i_idx] * float(weight[w_idx]);
                        }
                    }
                }
                out[oc * out_hw + oh * out_w + ow] = acc;
            }
        }
    }
}

static void batch_norm_infer(
    float buf[], int ch, int hw,
    const float gamma[], const float beta[],
    const float mean[], const float var[]) {
    for (int c = 0; c < ch; ++c) {
        const float inv_std = 1.f / std::sqrt(var[c] + BN_EPS);
        for (int i = 0; i < hw; ++i) {
#pragma HLS PIPELINE II = 1
            const int idx = c * hw + i;
            buf[idx] = gamma[c] * (buf[idx] - mean[c]) * inv_std + beta[c];
        }
    }
}

static void maxpool2x2(
    const float in[], int c, int in_h, int in_w,
    float out[], int out_h, int out_w) {
    const int in_hw = in_h * in_w;
    const int out_hw = out_h * out_w;

    for (int ch = 0; ch < c; ++ch) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
#pragma HLS PIPELINE II = 1
                float best = in[ch * in_hw + (oh * 2) * in_w + (ow * 2)];
                for (int kh = 0; kh < 2; ++kh) {
                    for (int kw = 0; kw < 2; ++kw) {
                        const float v = in[ch * in_hw + (oh * 2 + kh) * in_w + (ow * 2 + kw)];
                        if (v > best) best = v;
                    }
                }
                out[ch * out_hw + oh * out_w + ow] = best;
            }
        }
    }
}

static void linear(
    const float in[], int in_features,
    const float weight[], const float bias[],
    float out[], int out_features) {
    for (int o = 0; o < out_features; ++o) {
#pragma HLS PIPELINE II = 1
        float acc = bias[o];
        for (int i = 0; i < in_features; ++i) {
            acc += in[i] * weight[o * in_features + i];
        }
        out[o] = acc;
    }
}

void mnist_bnn_forward(
    const float input[CNN_IN_C][CNN_IN_H][CNN_IN_W],
    float output[CNN_NUM_CLASS]) {

#pragma HLS DATAFLOW disable

    static float buf_a[CNN_C1_OUT * CNN_H1 * CNN_W1];
    static float buf_b[CNN_C2_OUT * CNN_H2 * CNN_W2];
    static float buf_c[CNN_C3_OUT * CNN_H3 * CNN_W3];
    static float buf_p[CNN_C3_OUT * CNN_POOL_H * CNN_POOL_W];
    static float flat[CNN_FLAT];

#pragma HLS BIND_STORAGE variable = buf_a type = RAM_2P impl = BRAM
#pragma HLS BIND_STORAGE variable = buf_b type = RAM_2P impl = BRAM
#pragma HLS BIND_STORAGE variable = buf_c type = RAM_2P impl = BRAM

    static float in_bin[CNN_IN_C * CNN_IN_H * CNN_IN_W];
    for (int h = 0; h < CNN_IN_H; ++h) {
        for (int w = 0; w < CNN_IN_W; ++w) {
#pragma HLS PIPELINE II = 1
            in_bin[h * CNN_IN_W + w] = bin_sign(input[0][h][w]);
        }
    }

    // block 1: sign → conv → BN → sign
    bin_conv2d(in_bin, CNN_IN_C, CNN_IN_H, CNN_IN_W,
               &CONV1_WEIGHT[0][0][0][0], CNN_C1_OUT, CNN_K,
               buf_a, CNN_H1, CNN_W1);
    batch_norm_infer(buf_a, CNN_C1_OUT, CNN_H1 * CNN_W1,
                     BN1_WEIGHT, BN1_BIAS, BN1_RUNNING_MEAN, BN1_RUNNING_VAR);
    apply_bin_sign(buf_a, CNN_C1_OUT * CNN_H1 * CNN_W1);

    // block 2
    bin_conv2d(buf_a, CNN_C1_OUT, CNN_H1, CNN_W1,
               &CONV2_WEIGHT[0][0][0][0], CNN_C2_OUT, CNN_K,
               buf_b, CNN_H2, CNN_W2);
    batch_norm_infer(buf_b, CNN_C2_OUT, CNN_H2 * CNN_W2,
                     BN2_WEIGHT, BN2_BIAS, BN2_RUNNING_MEAN, BN2_RUNNING_VAR);
    apply_bin_sign(buf_b, CNN_C2_OUT * CNN_H2 * CNN_W2);

    // block 3
    bin_conv2d(buf_b, CNN_C2_OUT, CNN_H2, CNN_W2,
               &CONV3_WEIGHT[0][0][0][0], CNN_C3_OUT, CNN_K,
               buf_c, CNN_H3, CNN_W3);
    batch_norm_infer(buf_c, CNN_C3_OUT, CNN_H3 * CNN_W3,
                     BN3_WEIGHT, BN3_BIAS, BN3_RUNNING_MEAN, BN3_RUNNING_VAR);
    apply_bin_sign(buf_c, CNN_C3_OUT * CNN_H3 * CNN_W3);

    maxpool2x2(buf_c, CNN_C3_OUT, CNN_H3, CNN_W3, buf_p, CNN_POOL_H, CNN_POOL_W);

    for (int i = 0; i < CNN_FLAT; ++i) {
#pragma HLS PIPELINE II = 1
        flat[i] = buf_p[i];
    }

    linear(flat, CNN_FLAT, &FC_WEIGHT[0][0], FC_BIAS, output, CNN_NUM_CLASS);
}

int mnist_bnn_predict(
    const float input[CNN_IN_C][CNN_IN_H][CNN_IN_W],
    float output[CNN_NUM_CLASS]) {
    mnist_bnn_forward(input, output);
    int best = 0;
    for (int i = 1; i < CNN_NUM_CLASS; ++i) {
        if (output[i] > output[best]) best = i;
    }
    return best;
}
