#pragma once

#include <cstdint>

namespace fbgemm_gpu {

#if defined(USE_ROCM)
#if defined(__gfx1250__)
#define ROCM_WAVE32
#else
#define ROCM_WAVE64
#endif // defined(__gfx1250__)
#endif // defined(USE_ROCM)

#if defined(ROCM_WAVE64)
static constexpr int32_t kWarpSize = 64;
#else
static constexpr int32_t kWarpSize = 32;
#endif
} // namespace fbgemm_gpu
