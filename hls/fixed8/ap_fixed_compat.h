#pragma once

// Local C simulation on Mac/PC (no Vivado): float-backed ap_fixed stub.
// Vitis HLS:  -DUSE_VITIS_HLS  (uses Xilinx ap_fixed via ap_fixed.h wrapper)

#ifdef USE_VITIS_HLS
#include "ap_fixed.h"
#else

#include <cmath>
#include <cstdint>

#ifndef AP_RND
#define AP_RND 0
#endif
#ifndef AP_SAT
#define AP_SAT 0
#endif

template <int W, int I, int Q = AP_RND, int O = AP_SAT>
class ap_fixed {
public:
    ap_fixed() : v_(0.0f) {}
    ap_fixed(float x) : v_(x) {}
    ap_fixed(double x) : v_(static_cast<float>(x)) {}
    ap_fixed(int x) : v_(static_cast<float>(x)) {}

    operator float() const { return v_; }
    operator double() const { return static_cast<double>(v_); }

    ap_fixed& operator=(float x) {
        v_ = x;
        return *this;
    }

    ap_fixed& operator+=(const ap_fixed& o) {
        v_ += o.v_;
        return *this;
    }
    ap_fixed& operator-=(const ap_fixed& o) {
        v_ -= o.v_;
        return *this;
    }

    friend ap_fixed operator+(ap_fixed a, const ap_fixed& b) { return ap_fixed(a.v_ + b.v_); }
    friend ap_fixed operator-(ap_fixed a, const ap_fixed& b) { return ap_fixed(a.v_ - b.v_); }
    friend ap_fixed operator*(ap_fixed a, const ap_fixed& b) { return ap_fixed(a.v_ * b.v_); }
    friend ap_fixed operator/(ap_fixed a, const ap_fixed& b) { return ap_fixed(a.v_ / b.v_); }

    friend bool operator>(const ap_fixed& a, const ap_fixed& b) { return a.v_ > b.v_; }
    friend bool operator<(const ap_fixed& a, const ap_fixed& b) { return a.v_ < b.v_; }
    friend bool operator>=(const ap_fixed& a, const ap_fixed& b) { return a.v_ >= b.v_; }

private:
    float v_;
};

#endif
