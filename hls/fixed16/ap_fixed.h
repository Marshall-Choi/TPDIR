#pragma once

// CSIM: float stub (../common).  Vitis HLS: real ap_fixed via include_next.
#ifdef USE_VITIS_HLS
#include_next "ap_fixed.h"
#else
#include "../common/ap_fixed_compat.h"
#endif
