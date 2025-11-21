/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {
namespace {

constexpr int kGroupIndexWarpSize = kWarpSize;
constexpr int GROUP_INDEX_SELECT_UNROLL_FACTOR = 1;
constexpr int GROUP_INDEX_SELECT_COLS_PER_WARP =
    GROUP_INDEX_SELECT_UNROLL_FACTOR * kGroupIndexWarpSize;
constexpr int GROUP_INDEX_SELECT_LOG_COLS_PER_WARP =
    log2_calc<GROUP_INDEX_SELECT_COLS_PER_WARP>::value;

#ifdef USE_ROCM

constexpr int kGroupIndexLdsColsThreshold = 256;
constexpr int kGroupIndexLdsCacheSlots = 8;
constexpr size_t kGroupIndexLdsSharedMemBudgetBytes = 64 * 1024;

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

template <typename index_t, typename scalar_t, int COLS_PER_WARP>
constexpr size_t group_index_select_shared_mem_size() {
  constexpr size_t entry_bytes =
      sizeof(GroupIndexLdsCacheEntry<index_t, scalar_t, COLS_PER_WARP>);
  constexpr size_t total_bytes =
      entry_bytes * kGroupIndexLdsCacheSlots *
      (kMaxThreads / kGroupIndexWarpSize);
  return total_bytes <= kGroupIndexLdsSharedMemBudgetBytes ? total_bytes : 0;
}

template <bool UseIndexSelect, typename index_t, typename scalar_t, int COLS_PER_WARP>
constexpr size_t get_group_index_shared_mem_bytes() {
  if constexpr (UseIndexSelect) {
    return 0;
  } else {
    return group_index_select_shared_mem_size<
        index_t,
        scalar_t,
        COLS_PER_WARP>();
  }
}

template <
    typename index_t,
    typename scalar_t,
    bool USE_INDEX_SELECT,
    bool USE_VAR_COLS,
    int UNROLL_FACTOR,
    int COLS_PER_WARP,
    int LOG_COLS_PER_WARP>
__global__
__launch_bounds__(kMaxThreads) void group_index_select_or_add_2d_kernel(
    const int64_t* input_ptrs,
    const int64_t* output_ptrs,
    const int64_t* indices_ptrs,
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
        output[row * num_cols + i] = LDG(&input[idx * num_cols + i]);
      }
    }
  } else {
    constexpr int kCacheSlots = 2;
    constexpr bool kLdsCacheEnabled =
        group_index_select_shared_mem_size<
            index_t,
            scalar_t,
            COLS_PER_WARP>() > 0;
    using LdsCacheEntry =
        GroupIndexLdsCacheEntry<index_t, scalar_t, COLS_PER_WARP>;
    __shared__ int lds_slot_buffer[kMaxThreads / kGroupIndexWarpSize];
    __shared__ int lds_flush_slot_buffer[kMaxThreads / kGroupIndexWarpSize];
    __shared__ int lds_new_entry_buffer[kMaxThreads / kGroupIndexWarpSize];
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
      warp_cache = reinterpret_cast<LdsCacheEntry*>(group_index_shared_cache) +
          threadIdx.y * kGroupIndexLdsCacheSlots;
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
        (void)out_base;
        (void)num_cols;
        return;
      } else {
        for (int slot = 0; slot < kGroupIndexLdsCacheSlots; ++slot) {
          flush_lds_entry(slot, out_base, num_cols);
        }
      }
    };

    auto reset_lds_cache = [&]() {
      if constexpr (!kLdsCacheEnabled) {
        return;
      } else {
        for (int slot = threadIdx.x; slot < kGroupIndexLdsCacheSlots;
             slot += kGroupIndexWarpSize) {
          warp_cache[slot].valid = 0;
        }
        syncwarp();
      }
    };

    auto acquire_lds_slot = [&](index_t idx, int32_t col_offset) {
      if constexpr (!kLdsCacheEnabled) {
        (void)idx;
        (void)col_offset;
        return -1;
      } else {
        if (!warp_cache) {
          return -1;
        }
        if (threadIdx.x == 0) {
          uint32_t hash = static_cast<uint32_t>(idx) * 0x9e3779b1u ^
              static_cast<uint32_t>(col_offset);
          int candidate = hash & (kGroupIndexLdsCacheSlots - 1);
          int match_slot = -1;
          int empty_slot = -1;
          for (int attempt = 0; attempt < kGroupIndexLdsCacheSlots; ++attempt) {
            const int slot = (candidate + attempt) &
                (kGroupIndexLdsCacheSlots - 1);
            const auto& entry = warp_cache[slot];
            if (entry.valid && entry.idx == idx &&
                entry.col_offset == col_offset) {
              match_slot = slot;
              break;
            }
            if (!entry.valid && empty_slot == -1) {
              empty_slot = slot;
            }
          }
          int chosen_slot = match_slot >= 0 ? match_slot : empty_slot;
          if (chosen_slot < 0) {
            chosen_slot = candidate;
            lds_flush_slot_buffer[threadIdx.y] = chosen_slot;
          } else {
            lds_flush_slot_buffer[threadIdx.y] = -1;
          }
          lds_slot_buffer[threadIdx.y] = chosen_slot;
          lds_new_entry_buffer[threadIdx.y] = match_slot < 0;
        }
        syncwarp();
        int slot = lds_slot_buffer[threadIdx.y];
        const int flush_slot = lds_flush_slot_buffer[threadIdx.y];
        if (flush_slot >= 0) {
          flush_lds_entry(flush_slot, active_output_base, active_num_cols);
          if (threadIdx.x == 0) {
            warp_cache[flush_slot].valid = 0;
          }
          syncwarp();
          slot = flush_slot;
          lds_new_entry_buffer[threadIdx.y] = 1;
        }
        const bool initialize_slot = lds_new_entry_buffer[threadIdx.y];
        if (initialize_slot) {
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
      const bool member_uses_lds =
          kLdsCacheEnabled && (num_cols <= kGroupIndexLdsColsThreshold);
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

      const index_t idx = active_indices[row];

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

template <bool UseIndexSelect, typename index_t, typename scalar_t, int COLS_PER_WARP>
constexpr size_t get_group_index_shared_mem_bytes() {
  return 0;
}

template <
    typename index_t,
    typename scalar_t,
    bool USE_INDEX_SELECT,
    bool USE_VAR_COLS,
    int UNROLL_FACTOR,
    int COLS_PER_WARP,
    int LOG_COLS_PER_WARP>
__global__
__launch_bounds__(kMaxThreads) void group_index_select_or_add_2d_kernel(
    const int64_t* input_ptrs,
    const int64_t* output_ptrs,
    const int64_t* indices_ptrs,
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

#endif // USE_ROCM

} // namespace

int get_group_index_select_cols_per_warp() {
  return GROUP_INDEX_SELECT_COLS_PER_WARP;
}

DLL_PUBLIC void group_index_select_or_add_cuda(
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
  if (group_size == 0) {
    return;
  }

  at::cuda::OptionalCUDAGuard device_guard(device);

  uint32_t num_warps_per_threadblock = kMaxThreads / kGroupIndexWarpSize;
  uint32_t max_grid_size =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 8;
  uint32_t grid_size = std::min(
      cuda_calc_xblock_count(total_num_warps, num_warps_per_threadblock),
      max_grid_size);
  dim3 block_size(kGroupIndexWarpSize, num_warps_per_threadblock, 1);

#define INVOKE_GROUP_INDEX_SELECT_OR_ADD(USE_INDEX_SELECT_FLAG, USE_VAR_COLS_FLAG) \
  do { \
    constexpr size_t shared_mem_bytes = \
        get_group_index_shared_mem_bytes< \
            USE_INDEX_SELECT_FLAG, \
            index_t, \
            scalar_t, \
            GROUP_INDEX_SELECT_COLS_PER_WARP>(); \
    FBGEMM_LAUNCH_KERNEL( \
        (group_index_select_or_add_2d_kernel< \
            index_t, \
            scalar_t, \
            USE_INDEX_SELECT_FLAG, \
            USE_VAR_COLS_FLAG, \
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
        warp_offsets_group, \
        num_cols_group, \
        num_work_rows, \
        group_size); \
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
}

} // namespace fbgemm_gpu
