#pragma once

#ifdef USE_VITIS_HLS
#include_next "ap_fixed.h"
#else
#include "../common/ap_fixed_compat.h"
#endif
