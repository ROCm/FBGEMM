/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"
#if __has_include(<ATen/ATEN.h>)
#include <ATen/ATEN.h>
#elif __has_include(<ATen/ATen.h>)
#include <ATen/ATen.h>
#else
#error "ATen headers not found. Expected <ATen/ATEN.h> or <ATen/ATen.h>."
#endif
#include <vector>

using Tensor = at::Tensor;

namespace fbgemm_gpu {
namespace detail {

constexpr int kGroupIndexWarpSize = kWarpSize;
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
constexpr unsigned long long kGroupIndexFullWarpMask = 0xffffffffffffffffull;
#else
constexpr unsigned int kGroupIndexFullWarpMask = 0xffffffffu;
#endif
constexpr int GROUP_INDEX_SELECT_UNROLL_FACTOR = 2;
constexpr int GROUP_INDEX_SELECT_COLS_PER_WARP =
    GROUP_INDEX_SELECT_UNROLL_FACTOR * kGroupIndexWarpSize;
constexpr int GROUP_INDEX_SELECT_LOG_COLS_PER_WARP =
    log2_calc<GROUP_INDEX_SELECT_COLS_PER_WARP>::value;

enum class GroupIndexLdsStrategy : uint8_t {
  kDisabled = 0,
  kDirectMapped,
  kHashtable,
};

constexpr int kGroupIndexDirectMappedSlots = 32;
constexpr int kGroupIndexDirectMappedRowLimit = 48;
constexpr int kGroupIndexHashtableRowLimit = 192;

#ifdef USE_ROCM

constexpr int kGroupIndexLdsColsThreshold = 384;
constexpr int kGroupIndexLdsCacheSlots = 8;
constexpr size_t kGroupIndexLdsSharedMemBudgetBytes = 64 * 1024;

static_assert(
    (kGroupIndexDirectMappedSlots & (kGroupIndexDirectMappedSlots - 1)) == 0,
    "kGroupIndexDirectMappedSlots must be a power of two");
static_assert(
    (kGroupIndexLdsCacheSlots & (kGroupIndexLdsCacheSlots - 1)) == 0,
    "kGroupIndexLdsCacheSlots must be a power of two");

template <typename index_t, typename scalar_t, int COLS_PER_WARP>
struct alignas(16) GroupIndexLdsCacheEntry {
  index_t idx;
  int32_t col_offset;
  int32_t valid;
  scalar_t vals[COLS_PER_WARP];
};

template <
    typename index_t,
    typename scalar_t,
    int COLS_PER_WARP,
    bool UseDirectMapped>
constexpr size_t group_index_select_shared_mem_size() {
  constexpr int slot_count =
      UseDirectMapped ? kGroupIndexDirectMappedSlots : kGroupIndexLdsCacheSlots;
  constexpr size_t entry_bytes =
      sizeof(GroupIndexLdsCacheEntry<index_t, scalar_t, COLS_PER_WARP>);
  constexpr size_t total_bytes =
      entry_bytes * slot_count * (kMaxThreads / kGroupIndexWarpSize);
  return total_bytes <= kGroupIndexLdsSharedMemBudgetBytes ? total_bytes : 0;
}

template <
    bool UseIndexSelect,
    bool UseLdsCache,
    bool UseDirectMapped,
    typename index_t,
    typename scalar_t,
    int COLS_PER_WARP>
constexpr size_t get_group_index_shared_mem_bytes() {
  if constexpr (UseIndexSelect || !UseLdsCache) {
    return 0;
  }
  return group_index_select_shared_mem_size<
      index_t,
      scalar_t,
      COLS_PER_WARP,
      UseDirectMapped>();
}

template <
    typename index_t,
    typename scalar_t,
    bool USE_INDEX_SELECT,
    bool USE_VAR_COLS,
    bool USE_LDS_CACHE,
    bool USE_DIRECT_LDS,
    int UNROLL_FACTOR,
    int COLS_PER_WARP,
    int LOG_COLS_PER_WARP>
__global__
__launch_bounds__(kMaxThreads) void group_index_select_or_add_2d_kernel(
    const int64_t* input_ptrs,
    const int64_t* output_ptrs,
    const int64_t* indices_ptrs,
    const int64_t* row_order_ptrs,
    const int64_t* warp_offsets_group,
    const int32_t* num_cols_group,
    const int64_t num_work_rows,
    const int64_t group_size) {
  const auto total_num_warps = warp_offsets_group[group_size];
  extern __shared__ __align__(16) unsigned char group_index_shared_cache[];
  if (USE_INDEX_SELECT) {
    for (int64_t warp_id = threadIdx.y * gridDim.x + blockIdx.x;
         warp_id < total_num_warps;
         warp_id += gridDim.x * blockDim.y) {
      int32_t member_id, member_warp_id, num_cols, warps_per_row;
      if (USE_VAR_COLS) {
        __shared__ int member_ids[kMaxThreads / kGroupIndexWarpSize];
        if (threadIdx.x == 0) {
          binary_search_range(
              &member_ids[threadIdx.y],
              warp_offsets_group + 1,
              warp_id,
              group_size);
        }
        syncwarp();
        member_id = member_ids[threadIdx.y];
        num_cols = num_cols_group[member_id];
        warps_per_row = (num_cols + COLS_PER_WARP - 1) >> LOG_COLS_PER_WARP;
        member_warp_id = warp_id - warp_offsets_group[member_id];
      } else {
        num_cols = num_cols_group[0];
        warps_per_row = (num_cols + COLS_PER_WARP - 1) >> LOG_COLS_PER_WARP;
        member_id = warp_id / (warps_per_row * num_work_rows);
        member_warp_id = warp_id - (member_id * warps_per_row * num_work_rows);
      }
      const auto col_offset =
          ((member_warp_id % warps_per_row) << LOG_COLS_PER_WARP) +
          (threadIdx.x * UNROLL_FACTOR);
      const auto logical_row = member_warp_id / warps_per_row;
      const int32_t* member_row_order = nullptr;
      if (row_order_ptrs) {
        const int64_t ptr_val = row_order_ptrs[member_id];
        if (ptr_val) {
          member_row_order =
              reinterpret_cast<const int32_t*>(ptr_val);
        }
      }
      const int64_t row = member_row_order
          ? static_cast<int64_t>(member_row_order[logical_row])
          : logical_row;
      scalar_t* input =
          reinterpret_cast<scalar_t*>(input_ptrs[member_id]) + col_offset;
      scalar_t* output =
          reinterpret_cast<scalar_t*>(output_ptrs[member_id]) + col_offset;
      index_t* indices = reinterpret_cast<index_t*>(indices_ptrs[member_id]);
      index_t lane_idx = 0;
      if (threadIdx.x == 0) {
        lane_idx = indices[row];
      }
      const index_t idx = __shfl_sync(
          kGroupIndexFullWarpMask, lane_idx, 0, kGroupIndexWarpSize);
#pragma unroll
      for (int i = 0; i < UNROLL_FACTOR && col_offset + i < num_cols; i++) {
        output[row * num_cols + i] = LDG(&input[idx * num_cols + i]);
      }
    }
  } else {
    constexpr int kCacheSlots = 2;
    constexpr bool kLdsCacheEnabled =
        USE_LDS_CACHE &&
        group_index_select_shared_mem_size<
            index_t,
            scalar_t,
            COLS_PER_WARP,
            USE_DIRECT_LDS>() > 0;
    constexpr int kLdsSlotCount = USE_DIRECT_LDS
        ? kGroupIndexDirectMappedSlots
        : kGroupIndexLdsCacheSlots;
    using LdsCacheEntry =
        GroupIndexLdsCacheEntry<index_t, scalar_t, COLS_PER_WARP>;
    index_t cached_idx[kCacheSlots];
    scalar_t cached_vals[kCacheSlots][UNROLL_FACTOR];
    bool cached_valid[kCacheSlots];
#pragma unroll
    for (int slot = 0; slot < kCacheSlots; ++slot) {
      cached_valid[slot] = false;
    }
    int32_t active_member_id = -1;
    int32_t active_num_cols = 0;
    int32_t active_register_col_offset = -1;
    scalar_t* active_input_base = nullptr;
    scalar_t* active_output_base = nullptr;
    index_t* active_indices = nullptr;
    bool active_uses_lds = false;
    LdsCacheEntry* warp_cache = nullptr;
    if constexpr (kLdsCacheEnabled) {
      warp_cache =
          reinterpret_cast<LdsCacheEntry*>(group_index_shared_cache) +
          threadIdx.y * kLdsSlotCount;
    }
    const int lane_base = threadIdx.x * UNROLL_FACTOR;

    auto flush_register_cache = [&](scalar_t* out_base,
                                    int32_t num_cols,
                                    int32_t col_offset) {
      if (!out_base || col_offset < 0) {
        return;
      }
#pragma unroll
      for (int slot = 0; slot < kCacheSlots; ++slot) {
        if (!cached_valid[slot]) {
          continue;
        }
        const int64_t row_offset =
            static_cast<int64_t>(cached_idx[slot]) * num_cols;
#pragma unroll
        for (int j = 0; j < UNROLL_FACTOR; ++j) {
          const int32_t col = col_offset + j;
          if (col >= num_cols) {
            break;
          }
          gpuAtomicAddNoReturn(
              out_base + row_offset + col, cached_vals[slot][j]);
        }
        cached_valid[slot] = false;
      }
    };

    auto flush_lds_entry = [&](int slot, scalar_t* out_base, int32_t num_cols) {
      if constexpr (!kLdsCacheEnabled) {
        (void)slot;
        (void)out_base;
        (void)num_cols;
        return;
      } else {
        if (slot < 0 || !out_base) {
          return;
        }
        LdsCacheEntry& entry = warp_cache[slot];
        if (!entry.valid) {
          return;
        }
        const int64_t row_offset =
            static_cast<int64_t>(entry.idx) * num_cols;
#pragma unroll
        for (int j = 0; j < UNROLL_FACTOR; ++j) {
          const int col = lane_base + j;
          if (col >= COLS_PER_WARP) {
            break;
          }
          const int32_t global_col = entry.col_offset + col;
          if (global_col >= num_cols) {
            break;
          }
          gpuAtomicAddNoReturn(
              out_base + row_offset + global_col, entry.vals[col]);
          entry.vals[col] = static_cast<scalar_t>(0);
        }
        syncwarp();
        if (threadIdx.x == 0) {
          entry.valid = 0;
        }
        syncwarp();
      }
    };

    auto flush_lds_cache = [&](scalar_t* out_base, int32_t num_cols) {
      if constexpr (!kLdsCacheEnabled) {
        return;
      }
      for (int slot = 0; slot < kLdsSlotCount; ++slot) {
        flush_lds_entry(slot, out_base, num_cols);
      }
    };

    auto reset_lds_cache = [&]() {
      if constexpr (!kLdsCacheEnabled) {
        return;
      }
      for (int slot = threadIdx.x; slot < kLdsSlotCount;
           slot += kGroupIndexWarpSize) {
        warp_cache[slot].valid = 0;
      }
      syncwarp();
    };

    auto acquire_lds_slot = [&](index_t idx, int32_t col_offset) {
      if constexpr (!kLdsCacheEnabled) {
        return -1;
      } else if constexpr (USE_DIRECT_LDS) {
        const int slot =
            static_cast<int>(idx) & (kGroupIndexDirectMappedSlots - 1);
        LdsCacheEntry& entry = warp_cache[slot];
        const bool conflict = entry.valid && entry.idx != idx;
        if (conflict) {
          flush_lds_entry(slot, active_output_base, active_num_cols);
        }
        if (!entry.valid || conflict) {
#pragma unroll
          for (int j = 0; j < UNROLL_FACTOR; ++j) {
            const int col = lane_base + j;
            if (col >= COLS_PER_WARP) {
              break;
            }
            entry.vals[col] = static_cast<scalar_t>(0);
          }
          syncwarp();
          if (threadIdx.x == 0) {
            entry.idx = idx;
            entry.col_offset = col_offset;
            entry.valid = 1;
          }
          syncwarp();
        }
        return slot;
      } else {
        constexpr unsigned long long kFullMask = 0xffffffffffffffffull;
        constexpr int kSlotMask = kGroupIndexLdsCacheSlots - 1;
        uint32_t hash = 2166136261u;
        hash ^= static_cast<uint32_t>(idx);
        hash *= 16777619u;
        hash ^= static_cast<uint32_t>(col_offset);
        hash *= 16777619u;
        const int candidate = hash & kSlotMask;
        const bool lane_active = threadIdx.x < kGroupIndexLdsCacheSlots;
        const int lane_slot = lane_active
            ? ((candidate + threadIdx.x) & kSlotMask)
            : candidate;
        bool lane_match = false;
        bool lane_empty = false;
        if (lane_active) {
          const LdsCacheEntry& probed = warp_cache[lane_slot];
          lane_match = probed.valid && probed.idx == idx &&
              probed.col_offset == col_offset;
          lane_empty = !probed.valid;
        }
        const unsigned long long match_mask =
            __ballot_sync(kFullMask, lane_match);
        int slot_owner = 0;
        int slot = candidate;
        int initialize_flag = 0;
        if (match_mask) {
          slot_owner = __ffsll(match_mask) - 1;
          slot = __shfl_sync(kFullMask, lane_slot, slot_owner);
        } else {
          const unsigned long long empty_mask =
              __ballot_sync(kFullMask, lane_empty);
          if (empty_mask) {
            slot_owner = __ffsll(empty_mask) - 1;
            slot = __shfl_sync(kFullMask, lane_slot, slot_owner);
            initialize_flag = 1;
          } else {
            flush_lds_entry(slot, active_output_base, active_num_cols);
            initialize_flag = 1;
          }
        }
        slot = __shfl_sync(kFullMask, slot, slot_owner);
        initialize_flag =
            __shfl_sync(kFullMask, initialize_flag, slot_owner);
        if (initialize_flag) {
          if (threadIdx.x == 0) {
            warp_cache[slot].idx = idx;
            warp_cache[slot].col_offset = col_offset;
            warp_cache[slot].valid = 1;
          }
          syncwarp();
#pragma unroll
          for (int j = 0; j < UNROLL_FACTOR; ++j) {
            const int col = lane_base + j;
            if (col >= COLS_PER_WARP) {
              break;
            }
            warp_cache[slot].vals[col] = static_cast<scalar_t>(0);
          }
          syncwarp();
        }
        return slot;
      }
    };

    auto accumulate_into_lds = [&](index_t idx,
                                   int32_t col_offset,
                                   const scalar_t* local_vals) {
      if constexpr (!kLdsCacheEnabled) {
        (void)idx;
        (void)col_offset;
        (void)local_vals;
        return;
      } else {
        const int slot = acquire_lds_slot(idx, col_offset);
        if (slot < 0) {
          return;
        }
#pragma unroll
        for (int j = 0; j < UNROLL_FACTOR; ++j) {
          const int col = lane_base + j;
          if (col >= COLS_PER_WARP) {
            break;
          }
          const int32_t global_col = col_offset + col;
          if (global_col >= active_num_cols) {
            break;
          }
          warp_cache[slot].vals[col] += local_vals[j];
        }
      }
    };

    for (int64_t warp_id = threadIdx.y * gridDim.x + blockIdx.x;
         warp_id < total_num_warps;
         warp_id += gridDim.x * blockDim.y) {
      int32_t member_id, member_warp_id, num_cols, warps_per_row;
      if (USE_VAR_COLS) {
        __shared__ int member_ids[kMaxThreads / kGroupIndexWarpSize];
        if (threadIdx.x == 0) {
          binary_search_range(
              &member_ids[threadIdx.y],
              warp_offsets_group + 1,
              warp_id,
              group_size);
        }
        syncwarp();
        member_id = member_ids[threadIdx.y];
        num_cols = num_cols_group[member_id];
        warps_per_row = (num_cols + COLS_PER_WARP - 1) >> LOG_COLS_PER_WARP;
        member_warp_id = warp_id - warp_offsets_group[member_id];
      } else {
        num_cols = num_cols_group[0];
        warps_per_row = (num_cols + COLS_PER_WARP - 1) >> LOG_COLS_PER_WARP;
        member_id = warp_id / (warps_per_row * num_work_rows);
        member_warp_id = warp_id - (member_id * warps_per_row * num_work_rows);
      }
      const int64_t row = member_warp_id / warps_per_row;
      const int32_t col_offset =
          static_cast<int32_t>(((member_warp_id % warps_per_row)
                                << LOG_COLS_PER_WARP) +
                               (threadIdx.x * UNROLL_FACTOR));
      // Only enable LDS aggregation when a single warp fully covers the row
      // so we actually amortize the atomic traffic over duplicate indices.
      const bool member_uses_lds =
          kLdsCacheEnabled &&
          (num_work_rows <= (USE_DIRECT_LDS ? kGroupIndexDirectMappedRowLimit
                                            : kGroupIndexHashtableRowLimit)) &&
          (num_cols <= kGroupIndexLdsColsThreshold) && (warps_per_row == 1);
      const bool member_changed = member_id != active_member_id;
      const bool num_cols_changed =
          member_changed ? false : (num_cols != active_num_cols);
      if (member_changed || num_cols_changed ||
          member_uses_lds != active_uses_lds) {
        if (active_uses_lds) {
          flush_lds_cache(active_output_base, active_num_cols);
        } else {
          flush_register_cache(
              active_output_base, active_num_cols, active_register_col_offset);
        }
        active_member_id = member_id;
        active_num_cols = num_cols;
        active_input_base =
            reinterpret_cast<scalar_t*>(input_ptrs[member_id]);
        active_output_base =
            reinterpret_cast<scalar_t*>(output_ptrs[member_id]);
        active_indices =
            reinterpret_cast<index_t*>(indices_ptrs[member_id]);
        active_uses_lds = member_uses_lds;
        if (active_uses_lds) {
          reset_lds_cache();
          active_register_col_offset = -1;
        } else {
          active_register_col_offset = col_offset;
        }
      } else if (!active_uses_lds &&
                 col_offset != active_register_col_offset) {
        flush_register_cache(
            active_output_base, active_num_cols, active_register_col_offset);
        active_register_col_offset = col_offset;
      }

      if (col_offset >= active_num_cols) {
        continue;
      }

      index_t lane_idx = 0;
      if (threadIdx.x == 0) {
        lane_idx = active_indices[row];
      }
      const index_t idx = __shfl_sync(
          kGroupIndexFullWarpMask, lane_idx, 0, kGroupIndexWarpSize);

      scalar_t local_vals[UNROLL_FACTOR];
#pragma unroll
      for (int j = 0; j < UNROLL_FACTOR; ++j) {
        local_vals[j] = static_cast<scalar_t>(0);
      }
      const int64_t input_offset =
          static_cast<int64_t>(row) * active_num_cols + col_offset;
#pragma unroll
      for (int j = 0; j < UNROLL_FACTOR; ++j) {
        const int32_t col = col_offset + j;
        if (col >= active_num_cols) {
          break;
        }
        local_vals[j] = active_input_base[input_offset + j];
      }

      if (active_uses_lds) {
        accumulate_into_lds(idx, col_offset, local_vals);
        continue;
      }

      bool appended = false;
#pragma unroll
      for (int slot = 0; slot < kCacheSlots; ++slot) {
        if (cached_valid[slot] && cached_idx[slot] == idx) {
#pragma unroll
          for (int j = 0; j < UNROLL_FACTOR; ++j) {
            const int32_t col = col_offset + j;
            if (col >= active_num_cols) {
              break;
            }
            cached_vals[slot][j] += local_vals[j];
          }
          appended = true;
          break;
        }
      }

      if (!appended) {
        int slot_to_use = -1;
#pragma unroll
        for (int slot = 0; slot < kCacheSlots; ++slot) {
          if (!cached_valid[slot]) {
            slot_to_use = slot;
            break;
          }
        }
        if (slot_to_use == -1) {
          slot_to_use = 0;
          const int64_t row_offset =
              static_cast<int64_t>(cached_idx[slot_to_use]) *
              active_num_cols;
#pragma unroll
          for (int j = 0; j < UNROLL_FACTOR; ++j) {
            const int32_t col = col_offset + j;
            if (col >= active_num_cols) {
              break;
            }
            gpuAtomicAddNoReturn(
                active_output_base + row_offset + col,
                cached_vals[slot_to_use][j]);
          }
          cached_valid[slot_to_use] = false;
        }

        cached_idx[slot_to_use] = idx;
#pragma unroll
        for (int j = 0; j < UNROLL_FACTOR; ++j) {
          cached_vals[slot_to_use][j] = local_vals[j];
        }
        cached_valid[slot_to_use] = true;
      }
    }

    if (active_uses_lds) {
      flush_lds_cache(active_output_base, active_num_cols);
    } else {
      flush_register_cache(
          active_output_base, active_num_cols, active_register_col_offset);
    }
  }
}


#else // !USE_ROCM

template <bool UseIndexSelect, bool UseDirectMapped, typename index_t, typename scalar_t, int COLS_PER_WARP>
constexpr size_t get_group_index_shared_mem_bytes() {
  (void)UseDirectMapped;
  return 0;
}

template <
    typename index_t,
    typename scalar_t,
    bool USE_INDEX_SELECT,
    bool USE_VAR_COLS,
    bool /*USE_LDS_CACHE*/,
    bool /*USE_DIRECT_LDS*/,
    int UNROLL_FACTOR,
    int COLS_PER_WARP,
    int LOG_COLS_PER_WARP>
__global__
__launch_bounds__(kMaxThreads) void group_index_select_or_add_2d_kernel(
    const int64_t* input_ptrs,
    const int64_t* output_ptrs,
    const int64_t* indices_ptrs,
    const int64_t* row_order_ptrs,
    const int64_t* warp_offsets_group,
    const int32_t* num_cols_group,
    const int64_t num_work_rows,
    const int64_t group_size) {
  const auto total_num_warps = warp_offsets_group[group_size];
  int32_t num_cols = 0;
  int32_t warps_per_row = 0;

  if constexpr (!USE_VAR_COLS) {
    num_cols = num_cols_group[0];
    warps_per_row = (num_cols + COLS_PER_WARP - 1) >> LOG_COLS_PER_WARP;
  }

  for (int64_t warp_id = threadIdx.y * gridDim.x + blockIdx.x;
       warp_id < total_num_warps;
       warp_id += gridDim.x * blockDim.y) {
    int32_t member_id = 0;
    int32_t member_warp_id = 0;
    if constexpr (USE_VAR_COLS) {
      __shared__ int member_ids[kMaxThreads / kGroupIndexWarpSize];
      if (threadIdx.x == 0) {
        binary_search_range(
            &member_ids[threadIdx.y],
            warp_offsets_group + 1,
            warp_id,
            group_size);
      }
      syncwarp();
      member_id = member_ids[threadIdx.y];
      num_cols = num_cols_group[member_id];
      warps_per_row = (num_cols + COLS_PER_WARP - 1) >> LOG_COLS_PER_WARP;
      member_warp_id = warp_id - warp_offsets_group[member_id];
    } else {
      member_id = warp_id / (warps_per_row * num_work_rows);
      member_warp_id = warp_id - (member_id * warps_per_row * num_work_rows);
    }

    if (num_cols < COLS_PER_WARP) {
      // Optimized path for small embedding dimensions
      // Each warp processes 'rows_per_warp' rows
      int rows_per_warp = COLS_PER_WARP / num_cols;
      int64_t start_row = member_warp_id * rows_per_warp;
      
      // Since we are processing multiple rows within the warp, we need to
      // map each lane to a specific row, in addition to the column
      int local_row = (threadIdx.x * UNROLL_FACTOR) / num_cols; // the row ID within the set of rows handled by this warp
      int col = (threadIdx.x * UNROLL_FACTOR) % num_cols;
      int64_t current_row = start_row + local_row; // the actual row within the table processed by this lane

      // local_row may be out of bounds for the last few lanes in the warp
      // if [COLS_PER_WARP % num_cols != 0]
      // TODO: check if current_row < num_work_rows is necessary
      if (local_row < rows_per_warp && current_row < num_work_rows) {
        index_t* indices = reinterpret_cast<index_t*>(indices_ptrs[member_id]);
        index_t idx = indices[current_row];

        scalar_t* input_base = reinterpret_cast<scalar_t*>(input_ptrs[member_id]);
        scalar_t* output_base = reinterpret_cast<scalar_t*>(output_ptrs[member_id]);

#pragma unroll
        for (int i = 0; i < UNROLL_FACTOR && col + i < num_cols; i++) {
          if constexpr (USE_INDEX_SELECT) {
            output_base[current_row * num_cols + col] = 
                LDG(&input_base[idx * num_cols + col]);
          } else {
            gpuAtomicAddNoReturn(
                &output_base[idx * num_cols + col], 
                input_base[current_row * num_cols + col]);
          }
        }
      }
    } else {
      // Large embedding dimensions use >= 1 warp per row
      
      const auto row = member_warp_id / warps_per_row;
      const auto col_offset =
          ((member_warp_id % warps_per_row) << LOG_COLS_PER_WARP) +
          (threadIdx.x * UNROLL_FACTOR);
      
      scalar_t* input =
          reinterpret_cast<scalar_t*>(input_ptrs[member_id]) + col_offset;
      scalar_t* output =
          reinterpret_cast<scalar_t*>(output_ptrs[member_id]) + col_offset;

      index_t* indices = reinterpret_cast<index_t*>(indices_ptrs[member_id]);
      const index_t idx = indices[row];

#pragma unroll
      for (int i = 0; i < UNROLL_FACTOR && col_offset + i < num_cols; i++) {
        if constexpr (USE_INDEX_SELECT) {
          output[row * num_cols + i] = LDG(&input[idx * num_cols + i]);
        } else {
          gpuAtomicAddNoReturn(
              &output[idx * num_cols + i], input[row * num_cols + i]);
        }
      }
    }
  }
}

#endif // USE_ROCM

void group_index_select_or_add_cuda_impl(
    const int64_t* input_ptrs,
    const int64_t* output_ptrs,
    const int64_t* indices_ptrs,
    const int64_t* row_order_ptrs,
    const int64_t* warp_offsets_group,
    const int32_t* num_cols_group,
    const c10::ScalarType& input_scalar_type,
    const c10::ScalarType& indices_scalar_type,
    const c10::DeviceIndex& device,
    const int num_work_rows,
    const int64_t total_num_warps,
    const int group_size,
    const bool use_index_select,
    const bool use_var_cols) {
  if (group_size == 0) {
    return;
  }
  at::cuda::OptionalCUDAGuard device_guard(device);
#ifdef USE_ROCM
  GroupIndexLdsStrategy lds_strategy = GroupIndexLdsStrategy::kDisabled;
  if (!use_index_select && group_size > 0) {
    auto num_cols_host = at::empty(
        {group_size},
        at::TensorOptions().dtype(at::kInt).pinned_memory(true));
    auto stream = at::cuda::getCurrentCUDAStream();
    auto memcpy_status = cudaMemcpyAsync(
        num_cols_host.data_ptr<int32_t>(),
        num_cols_group,
        sizeof(int32_t) * group_size,
        cudaMemcpyDeviceToHost,
        stream.stream());
    TORCH_CHECK(
        memcpy_status == cudaSuccess,
        "Failed to copy num_cols_group for LDS heuristic");
    TORCH_CHECK(
        cudaStreamSynchronize(stream.stream()) == cudaSuccess,
        "Failed to sync LDS heuristic copy");
    const int32_t* num_cols_host_ptr = num_cols_host.data_ptr<int32_t>();
    const int cols_per_warp = get_group_index_select_cols_per_warp();
    bool lds_candidate_found = false;
    for (int i = 0; i < group_size; ++i) {
      const int num_cols = num_cols_host_ptr[i];
      const int warps_per_row =
          (num_cols + cols_per_warp - 1) / cols_per_warp;
      if (num_cols <= kGroupIndexLdsColsThreshold && warps_per_row == 1) {
        lds_candidate_found = true;
        break;
      }
    }
    if (lds_candidate_found) {
      if (num_work_rows <= kGroupIndexDirectMappedRowLimit) {
        lds_strategy = GroupIndexLdsStrategy::kDirectMapped;
      } else if (num_work_rows <= kGroupIndexHashtableRowLimit) {
        lds_strategy = GroupIndexLdsStrategy::kHashtable;
      }
    }
  }
#else
  constexpr GroupIndexLdsStrategy lds_strategy = GroupIndexLdsStrategy::kDisabled;
#endif
  uint32_t num_warps_per_threadblock = kMaxThreads / kGroupIndexWarpSize;
  dim3 block_size(kGroupIndexWarpSize, num_warps_per_threadblock, 1);
  const auto* device_prop = at::cuda::getCurrentDeviceProperties();
  const int block_threads = block_size.x * block_size.y * block_size.z;
  const int max_blocks_per_sm = std::max(
      1,
      device_prop->maxThreadsPerMultiProcessor / block_threads);
  const uint32_t requested_grid =
      cuda_calc_xblock_count(total_num_warps, num_warps_per_threadblock);
  const int occupancy_multiplier = use_index_select ? 6 : 8;
  const uint32_t occupancy_cap = std::max(
      1,
      max_blocks_per_sm * device_prop->multiProcessorCount *
          occupancy_multiplier);
  uint32_t grid_size = std::min(requested_grid, occupancy_cap);
  grid_size =
      std::max<uint32_t>(grid_size, device_prop->multiProcessorCount);

#define INVOKE_GROUP_INDEX_SELECT_OR_ADD_WITH_STRATEGY(USE_INDEX_SELECT_FLAG, USE_VAR_COLS_FLAG, USE_LDS_CACHE_FLAG, USE_DIRECT_LDS_FLAG) \
  do { \
    constexpr size_t shared_mem_bytes = \
        get_group_index_shared_mem_bytes< \
            USE_INDEX_SELECT_FLAG, \
            USE_LDS_CACHE_FLAG, \
            USE_DIRECT_LDS_FLAG, \
            index_t, \
            scalar_t, \
            GROUP_INDEX_SELECT_COLS_PER_WARP>(); \
    FBGEMM_LAUNCH_KERNEL( \
        (group_index_select_or_add_2d_kernel< \
            index_t, \
            scalar_t, \
            USE_INDEX_SELECT_FLAG, \
            USE_VAR_COLS_FLAG, \
            USE_LDS_CACHE_FLAG, \
            USE_DIRECT_LDS_FLAG, \
            GROUP_INDEX_SELECT_UNROLL_FACTOR, \
            GROUP_INDEX_SELECT_COLS_PER_WARP, \
            GROUP_INDEX_SELECT_LOG_COLS_PER_WARP>), \
        grid_size, \
        block_size, \
        shared_mem_bytes, \
        at::cuda::getCurrentCUDAStream(), \
        input_ptrs, \
        output_ptrs, \
        indices_ptrs, \
        row_order_ptrs, \
        warp_offsets_group, \
        num_cols_group, \
        num_work_rows, \
        group_size); \
  } while (0)

#define INVOKE_GROUP_INDEX_SELECT_OR_ADD(USE_INDEX_SELECT_FLAG, USE_VAR_COLS_FLAG) \
  do { \
    if (lds_strategy == GroupIndexLdsStrategy::kDirectMapped) { \
      INVOKE_GROUP_INDEX_SELECT_OR_ADD_WITH_STRATEGY( \
          USE_INDEX_SELECT_FLAG, USE_VAR_COLS_FLAG, true, true); \
    } else if (lds_strategy == GroupIndexLdsStrategy::kHashtable) { \
      INVOKE_GROUP_INDEX_SELECT_OR_ADD_WITH_STRATEGY( \
          USE_INDEX_SELECT_FLAG, USE_VAR_COLS_FLAG, true, false); \
    } else { \
      INVOKE_GROUP_INDEX_SELECT_OR_ADD_WITH_STRATEGY( \
          USE_INDEX_SELECT_FLAG, USE_VAR_COLS_FLAG, false, false); \
    } \
  } while (0)

  AT_DISPATCH_INDEX_TYPES(
      indices_scalar_type, "group_index_select_2d_wrapper_1", [&] {
        FBGEMM_DISPATCH_FLOATING_TYPES(
            input_scalar_type, "group_index_select_2d_wrapper_2", [&] {
              if (use_index_select) {
                if (use_var_cols) {
                  INVOKE_GROUP_INDEX_SELECT_OR_ADD(true, true);
                } else {
                  INVOKE_GROUP_INDEX_SELECT_OR_ADD(true, false);
                }
              } else {
                if (use_var_cols) {
                  INVOKE_GROUP_INDEX_SELECT_OR_ADD(false, true);
                } else {
                  INVOKE_GROUP_INDEX_SELECT_OR_ADD(false, false);
                }
              }
            });
      });

#undef INVOKE_GROUP_INDEX_SELECT_OR_ADD
#undef INVOKE_GROUP_INDEX_SELECT_OR_ADD_WITH_STRATEGY
} // namespace detail
}

int get_group_index_select_cols_per_warp() {
  return detail::GROUP_INDEX_SELECT_COLS_PER_WARP;
}

void group_index_select_or_add_with_row_order_cuda(
    const int64_t* input_ptrs,
    const int64_t* output_ptrs,
    const int64_t* indices_ptrs,
    const int64_t* row_order_ptrs,
    const int64_t* warp_offsets_group,
    const int32_t* num_cols_group,
    const c10::ScalarType& input_scalar_type,
    const c10::ScalarType& indices_scalar_type,
    const c10::DeviceIndex& device,
    const int num_work_rows,
    const int64_t total_num_warps,
    const int group_size,
    const bool use_index_select,
    const bool use_var_cols) {
  detail::group_index_select_or_add_cuda_impl(
      input_ptrs,
      output_ptrs,
      indices_ptrs,
      row_order_ptrs,
      warp_offsets_group,
      num_cols_group,
      input_scalar_type,
      indices_scalar_type,
      device,
      num_work_rows,
      total_num_warps,
      group_size,
      use_index_select,
      use_var_cols);
}

void group_index_select_or_add_cuda(
    const int64_t* input_ptrs,
    const int64_t* output_ptrs,
    const int64_t* indices_ptrs,
    const int64_t* warp_offsets_group,
    const int32_t* num_cols_group,
    const c10::ScalarType& input_scalar_type,
    const c10::ScalarType& indices_scalar_type,
    const c10::DeviceIndex& device,
    const int num_work_rows,
    const int64_t total_num_warps,
    const int group_size,
    const bool use_index_select,
    const bool use_var_cols) {
  detail::group_index_select_or_add_cuda_impl(
      input_ptrs,
      output_ptrs,
      indices_ptrs,
      /*row_order_ptrs=*/nullptr,
      warp_offsets_group,
      num_cols_group,
      input_scalar_type,
      indices_scalar_type,
      device,
      num_work_rows,
      total_num_warps,
      group_size,
      use_index_select,
      use_var_cols);
}

} // namespace fbgemm_gpu