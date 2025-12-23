/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_runtime.h>

#include "fbgemm_gpu/utils/cuda_prelude.cuh"

namespace fbgemm_gpu {

#if defined(USE_ROCM)
// Warp-synchronous upper_bound used when all lanes share the same target.
template <typename scalar_t, int kLogicalWarpSize = kWarpSize>
__device__ __forceinline__ void warp_upper_bound(
    int* found,
    int* cached_boundary,
    const scalar_t* arr,
    const scalar_t target,
    const int num_entries) {
  const auto active_mask = __activemask();
  using mask_t = std::remove_const_t<decltype(active_mask)>;

  constexpr int kHardwareWarpSize = kWarpSize;
  constexpr int kMaskBits = sizeof(mask_t) * 8;

  const int hardware_lane = __lane_id();
  const int logical_lane = hardware_lane % kLogicalWarpSize;
  const int logical_warp_id = hardware_lane / kLogicalWarpSize;

  mask_t logical_mask = mask_t(0);
  if constexpr (kLogicalWarpSize >= kMaskBits) {
    logical_mask = active_mask;
  } else {
    const mask_t group_bits = (mask_t(1) << kLogicalWarpSize) - 1;
    const mask_t group_mask = group_bits << (logical_warp_id * kLogicalWarpSize);
    logical_mask = group_mask & active_mask;
  }
  if (!logical_mask) {
    logical_mask = active_mask;
  }

  int result = -1;
  int cached_result = *cached_boundary;
  for (int base = 0; base < num_entries; base += kLogicalWarpSize) {
    const int idx = base + logical_lane;
    const bool valid = idx < num_entries;
    const scalar_t val = valid ? arr[idx] : scalar_t(0);
    const mask_t ballot = __ballot_sync(active_mask, valid && val > target);
    const mask_t logical_ballot = ballot & logical_mask;
    if (logical_ballot) {
#if defined(__HIP_PLATFORM_AMD__)
      const int first_lane_hw = __ffsll(static_cast<long long>(logical_ballot)) - 1;
#else
      const int first_lane_hw = __ffs(static_cast<int>(logical_ballot)) - 1;
#endif
      const int first_lane = first_lane_hw - logical_warp_id * kLogicalWarpSize;
      result = base + first_lane;
      cached_result = __shfl_sync(active_mask, val, first_lane_hw);
      break;
    }
  }

  *found = result;
  *cached_boundary = cached_result;
}
#endif

template <typename scalar_t>
__device__ __forceinline__ void binary_search_range(
    int* found,
    const scalar_t* arr,
    const scalar_t target,
    const int num_entries) {
  const int last_entry = num_entries - 1;
  int start = 0, end = last_entry;
  int found_ = -1;
  while (start <= end) {
    int mid = start + (end - start) / 2;
    scalar_t mid_offset = arr[mid];
    if (target == mid_offset) {
      if (mid != last_entry && target != arr[last_entry]) {
        // Do linear scan in case of duplicate data (We assume that the
        // number of duplicates is small.  This can we very bad if the
        // number of duplicates is large)
        for (int i = mid + 1; i < num_entries; i++) {
          if (target != arr[i]) {
            found_ = i;
            break;
          }
        }
      }
      break;
    } else if (target < mid_offset) {
      if (mid == 0) {
        found_ = 0;
        break;
      } else if (mid - 1 >= 0 && target > arr[mid - 1]) {
        found_ = mid;
        break;
      }
      end = mid - 1;
    } else {
      if (mid + 1 <= last_entry && target < arr[mid + 1]) {
        found_ = mid + 1;
        break;
      }
      start = mid + 1;
    }
  }
  *found = found_;
}

} // namespace fbgemm_gpu
