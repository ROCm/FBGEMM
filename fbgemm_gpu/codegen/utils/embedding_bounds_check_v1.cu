/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/utils/embedding_bounds_check_common.cuh"

#include <cstdio>

namespace {
constexpr int64_t kNumThreadsPerBlock = 256;
constexpr int64_t kWarpsPerBlock =
    kNumThreadsPerBlock / fbgemm_gpu::kWarpSize;
constexpr int64_t kInvalidInfoSize = 10;
constexpr int64_t kOffsetInfoSize = 6;
constexpr int64_t kLastOffsetInfoSize = 3;
#if defined(USE_ROCM)
constexpr auto kWarpSyncMask = fbgemm_gpu::kFullWarpMask;
#else
constexpr unsigned kWarpSyncMask = 0xffffffffu;
#endif
} // namespace

template <typename index_t, bool vbe, BoundsCheckMode bounds_check_mode>
__global__ __launch_bounds__(kMaxThreads) void bounds_check_indices_kernel_v1(
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        rows_per_table,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    const int32_t* const B_offsets,
    [[maybe_unused]] pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        warning,
    FixedDivisor fd,
    int64_t* const per_warp_warning_count,
    int64_t* const per_warp_invalid_info,
    int64_t* const per_warp_offset_info,
    int64_t* const last_offset_info,
    TORCH_DSA_KERNEL_ARGS) {
  const int32_t T = rows_per_table.size(0);
  const int32_t total_B = offsets.size(0) - 1;
  const index_t num_indices = indices.size(0);
  const int32_t warp_linear_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t lane_id = threadIdx.x;

  const int32_t logical_b_t = warp_linear_id;

  if constexpr (!vbe) {
    if (logical_b_t >= total_B) {
      return;
    }
  }

  int32_t t = 0;
  int32_t b = 0;
  fd.DivMod(logical_b_t, &t, &b);

  int32_t physical_b_t = logical_b_t;
  int32_t B_local = 0;

  if constexpr (vbe) {
    if (t >= T) {
      return;
    }
    const auto B_start = B_offsets[t];
    const auto B_end = B_offsets[t + 1];
    B_local = B_end - B_start;
    if (b >= B_local) {
      return;
    }
    physical_b_t = B_start + b;
  } else {
    B_local = (T == 0 ? 0 : total_B / T);
    if (b >= B_local) {
      return;
    }
  }

  const auto num_rows = rows_per_table[t];

  [[maybe_unused]] int64_t warp_warning_accum = 0;
  [[maybe_unused]] int64_t warp_invalid_flag = 0;
  [[maybe_unused]] int64_t warp_invalid_b = 0;
  [[maybe_unused]] int64_t warp_invalid_t = 0;
  [[maybe_unused]] int64_t warp_invalid_bag_element = 0;
  [[maybe_unused]] int64_t warp_invalid_idx = 0;
  [[maybe_unused]] int64_t warp_invalid_num_rows = 0;
  [[maybe_unused]] int64_t warp_invalid_indices_start = 0;
  [[maybe_unused]] int64_t warp_invalid_indices_end = 0;
  [[maybe_unused]] int64_t warp_invalid_B = 0;
  [[maybe_unused]] int64_t warp_invalid_b_t = 0;

  [[maybe_unused]] int64_t warp_offset_flag = 0;
  [[maybe_unused]] int64_t warp_offset_b = 0;
  [[maybe_unused]] int64_t warp_offset_t = 0;
  [[maybe_unused]] int64_t warp_offset_indices_start = 0;
  [[maybe_unused]] int64_t warp_offset_indices_end = 0;
  [[maybe_unused]] int64_t warp_offset_num_indices = 0;

  if (logical_b_t == 0 && lane_id == 0) {
    if constexpr (bounds_check_mode == BoundsCheckMode::FATAL) {
      CUDA_KERNEL_ASSERT2(num_indices == offsets[total_B]);
    } else if constexpr (bounds_check_mode == BoundsCheckMode::WARNING) {
      const index_t current_last = offsets[total_B];
      if (num_indices != current_last) {
        if (last_offset_info != nullptr) {
          last_offset_info[0] = 1;
          last_offset_info[1] = static_cast<int64_t>(current_last);
          last_offset_info[2] = static_cast<int64_t>(num_indices);
        }
        warp_warning_accum += 1;
        offsets[total_B] = num_indices;
      }
    } else {
      if (num_indices != offsets[total_B]) {
        offsets[total_B] = num_indices;
      }
    }
  }

  auto indices_start = offsets[physical_b_t];
  auto indices_end = offsets[physical_b_t + 1];

  if constexpr (bounds_check_mode == BoundsCheckMode::FATAL) {
    CUDA_KERNEL_ASSERT2(indices_start >= 0);
    CUDA_KERNEL_ASSERT2(indices_start <= indices_end);
    CUDA_KERNEL_ASSERT2(indices_end <= num_indices);
  } else if constexpr (bounds_check_mode == BoundsCheckMode::WARNING) {
    const auto original_start = indices_start;
    const auto original_end = indices_end;
    if (indices_start < 0 || indices_start > indices_end ||
        indices_end > num_indices) {
      if (lane_id == 0) {
        warp_warning_accum += 1;
        if (warp_offset_flag == 0) {
          warp_offset_flag = 1;
          warp_offset_b = b;
          warp_offset_t = t;
          warp_offset_indices_start = static_cast<int64_t>(original_start);
          warp_offset_indices_end = static_cast<int64_t>(original_end);
          warp_offset_num_indices = static_cast<int64_t>(num_indices);
        }
      }
      adjust_offset_kernel(
          indices_start,
          indices_end,
          num_indices,
          &offsets[physical_b_t],
          &offsets[physical_b_t + 1]);
    }
  } else {
    adjust_offset_kernel(
        indices_start,
        indices_end,
        num_indices,
        &offsets[physical_b_t],
        &offsets[physical_b_t + 1]);
  }

  const auto L = indices_end - indices_start;

  if constexpr (bounds_check_mode == BoundsCheckMode::FATAL) {
    for (index_t i = static_cast<index_t>(lane_id); i < L;
         i += static_cast<index_t>(fbgemm_gpu::kWarpSize)) {
      const auto idx = indices[indices_start + i];
      if (idx == static_cast<index_t>(-1)) {
        continue;
      }
      CUDA_KERNEL_ASSERT2(
          idx >= 0 && "Failed idx >= 0 in bounds_check_indices");
      CUDA_KERNEL_ASSERT2(
          idx < num_rows && "Failed idx < num_rows in bounds_check_indices");
    }
  } else if constexpr (bounds_check_mode == BoundsCheckMode::WARNING) {
    int64_t thread_warning = 0;
    bool thread_has_invalid = false;
    index_t thread_invalid_i = static_cast<index_t>(-1);
    index_t thread_invalid_idx = static_cast<index_t>(0);

    for (index_t i = static_cast<index_t>(lane_id); i < L;
         i += static_cast<index_t>(fbgemm_gpu::kWarpSize)) {
      const auto idx = indices[indices_start + i];
      if (idx == static_cast<index_t>(-1)) {
        continue;
      }
      if (idx < 0 || idx >= num_rows) {
        if (!thread_has_invalid) {
          thread_has_invalid = true;
          thread_invalid_i = i;
          thread_invalid_idx = idx;
        }
        indices[indices_start + i] = 0;
        thread_warning += 1;
      }
    }

#if defined(USE_ROCM)
    const unsigned long long invalid_mask =
        __ballot_sync(kWarpSyncMask, thread_has_invalid);
#else
    const unsigned invalid_mask =
        __ballot_sync(kWarpSyncMask, thread_has_invalid);
#endif
    if (invalid_mask != 0 && lane_id == 0) {
#if defined(USE_ROCM)
      const int32_t first_lane = __ffsll(invalid_mask) - 1;
#else
      const int32_t first_lane = __ffs(invalid_mask) - 1;
#endif
      warp_invalid_flag = 1;
      warp_invalid_b = b;
      warp_invalid_t = t;
      warp_invalid_bag_element = static_cast<int64_t>(__shfl_sync(
          kWarpSyncMask,
          static_cast<int64_t>(thread_invalid_i),
          first_lane));
      warp_invalid_idx = static_cast<int64_t>(__shfl_sync(
          kWarpSyncMask,
          static_cast<int64_t>(thread_invalid_idx),
          first_lane));
      warp_invalid_num_rows = static_cast<int64_t>(num_rows);
      warp_invalid_indices_start = static_cast<int64_t>(indices_start);
      warp_invalid_indices_end = static_cast<int64_t>(indices_end);
      warp_invalid_b_t = physical_b_t;
      if constexpr (vbe) {
        warp_invalid_B = static_cast<int64_t>(B_local);
      } else {
        warp_invalid_B = static_cast<int64_t>(B_local);
      }
    }

    int64_t warp_sum = thread_warning;
    for (int offset = fbgemm_gpu::kWarpSize / 2; offset > 0; offset /= 2) {
      warp_sum += __shfl_down_sync(kWarpSyncMask, warp_sum, offset);
    }
    if (lane_id == 0) {
      warp_warning_accum += warp_sum;
    }
  } else {
    for (index_t i = static_cast<index_t>(lane_id); i < L;
         i += static_cast<index_t>(fbgemm_gpu::kWarpSize)) {
      const auto idx = indices[indices_start + i];
      if (idx == static_cast<index_t>(-1)) {
        continue;
      }
      if (idx < 0 || idx >= num_rows) {
        indices[indices_start + i] = 0;
      }
    }
  }

  if constexpr (bounds_check_mode == BoundsCheckMode::WARNING) {
    if (lane_id == 0) {
      if (per_warp_warning_count != nullptr) {
        per_warp_warning_count[warp_linear_id] = warp_warning_accum;
      }
      if (per_warp_invalid_info != nullptr) {
        int64_t* const slot =
            per_warp_invalid_info + warp_linear_id * kInvalidInfoSize;
        slot[0] = warp_invalid_flag;
        slot[1] = warp_invalid_b;
        slot[2] = warp_invalid_t;
        slot[3] = warp_invalid_bag_element;
        slot[4] = warp_invalid_idx;
        slot[5] = warp_invalid_num_rows;
        slot[6] = warp_invalid_indices_start;
        slot[7] = warp_invalid_indices_end;
        slot[8] = warp_invalid_B;
        slot[9] = warp_invalid_b_t;
      }
      if (per_warp_offset_info != nullptr) {
        int64_t* const slot =
            per_warp_offset_info + warp_linear_id * kOffsetInfoSize;
        slot[0] = warp_offset_flag;
        slot[1] = warp_offset_b;
        slot[2] = warp_offset_t;
        slot[3] = warp_offset_indices_start;
        slot[4] = warp_offset_indices_end;
        slot[5] = warp_offset_num_indices;
      }
    }
  }
}

void _bounds_check_indices_cuda_v1(
    Tensor& rows_per_table,
    Tensor& indices,
    Tensor& offsets,
    BoundsCheckMode bounds_check_mode,
    Tensor& warning,
    const std::optional<Tensor>& weights,
    const std::optional<Tensor>& B_offsets,
    int64_t max_B,
    const std::optional<Tensor>& /*b_t_map*/,
    int32_t /*info_B_num_bits*/,
    uint32_t /*info_B_mask*/,
    int64_t T,
    int64_t B,
    int64_t /*total_B*/,
    bool vbe,
    bool prefetch_pipeline) {
  TORCH_CHECK(
      !prefetch_pipeline,
      "bounds_check_indices_v1 does not support prefetch_pipeline=true");

  CUDA_DEVICE_GUARD(rows_per_table);

  if (bounds_check_mode == BoundsCheckMode::WARNING) {
    warning.zero_();
  }

  const auto max_B_ = vbe ? max_B : B;
  auto grid_dim = div_round_up(
      max_B_ * T, kNumThreadsPerBlock / fbgemm_gpu::kWarpSize);

  const auto num_warps =
      static_cast<int64_t>(grid_dim) * kWarpsPerBlock;

  Tensor per_warp_warning_counts;
  Tensor per_warp_invalid_info;
  Tensor per_warp_offset_info;
  Tensor last_offset_info;

  int64_t* warning_counts_ptr = nullptr;
  int64_t* invalid_info_ptr = nullptr;
  int64_t* offset_info_ptr = nullptr;
  int64_t* last_offset_ptr = nullptr;

  if (bounds_check_mode == BoundsCheckMode::WARNING) {
    auto buffer_opts = indices.options().dtype(at::kLong);
    per_warp_warning_counts = at::zeros({num_warps}, buffer_opts);
    per_warp_invalid_info =
        at::zeros({num_warps, kInvalidInfoSize}, buffer_opts);
    per_warp_offset_info =
        at::zeros({num_warps, kOffsetInfoSize}, buffer_opts);
    last_offset_info = at::zeros({kLastOffsetInfoSize}, buffer_opts);

    warning_counts_ptr = per_warp_warning_counts.data_ptr<int64_t>();
    invalid_info_ptr = per_warp_invalid_info.data_ptr<int64_t>();
    offset_info_ptr = per_warp_offset_info.data_ptr<int64_t>();
    last_offset_ptr = last_offset_info.data_ptr<int64_t>();
  }

#define INVOKE_BOUNDS_CHECK_INDICES(MODE)                                        if (bounds_check_mode == MODE) {                                                 AT_DISPATCH_INDEX_TYPES(                                                           indices.scalar_type(), "bounds_check_indices_cuda_v1", [&] {                      const auto bounds_check_kernel =                                                   (vbe ? bounds_check_indices_kernel_v1<index_t, true, MODE>                          : bounds_check_indices_kernel_v1<index_t, false, MODE>);              FBGEMM_LAUNCH_DSA_KERNEL(                                                          bounds_check_kernel,                                                           grid_dim,                                                                      dim3(                                                                              fbgemm_gpu::kWarpSize,                                                         kNumThreadsPerBlock / fbgemm_gpu::kWarpSize),                              0,                                                                             at::cuda::getCurrentCUDAStream(),                                              PTA_B(rows_per_table, int64_t, 1, 32),                                         PTA_B(indices, index_t, 1, 32),                                                PTA_B(offsets, index_t, 1, 32),                                                vbe ? B_offsets.value().data_ptr<int32_t>() : nullptr,                         PTA_B(warning, int64_t, 1, 32),                                                FixedDivisor(max_B_),                                                          warning_counts_ptr,                                                            invalid_info_ptr,                                                              offset_info_ptr,                                                              last_offset_ptr);                                                        });                                                                      }

  INVOKE_BOUNDS_CHECK_INDICES(BoundsCheckMode::FATAL)
  INVOKE_BOUNDS_CHECK_INDICES(BoundsCheckMode::WARNING)
  INVOKE_BOUNDS_CHECK_INDICES(BoundsCheckMode::IGNORE)

#undef INVOKE_BOUNDS_CHECK_INDICES

  if (bounds_check_mode == BoundsCheckMode::WARNING) {
    auto total_warning_tensor = per_warp_warning_counts.sum();
    const auto total_warning = total_warning_tensor.item<int64_t>();
    warning.fill_(total_warning);

    const auto last_offset_host = last_offset_info.to(at::kCPU);
    const auto offset_info_host = per_warp_offset_info.to(at::kCPU);
    const auto invalid_info_host = per_warp_invalid_info.to(at::kCPU);

    const auto total_B = offsets.size(0) - 1;

    const int64_t* last_offset_ptr_host =
        last_offset_host.data_ptr<int64_t>();
    if (last_offset_ptr_host[0]) {
      const auto prev_last = last_offset_ptr_host[1];
      const auto corrected_last = last_offset_ptr_host[2];
      std::printf(
          "EmbeddingBoundsCheck (VBE %s): the last element in offsets is "
          "incorrect for total batch size %s: %d, total table num T: %d, "
          " last element in offsets: %lld, indices size: %lld. "
          " Setting the last element in offsets to be indices size.\n",
          vbe ? "true" : "false",
          vbe ? "total_B" : "B",
          vbe ? static_cast<int>(total_B) : static_cast<int>(B),
          static_cast<int>(T),
          static_cast<long long>(prev_last),
          static_cast<long long>(corrected_last));
    }

    const int64_t* offset_ptr_host = offset_info_host.data_ptr<int64_t>();
    for (int64_t warp = 0; warp < num_warps; ++warp) {
      if (offset_ptr_host[warp * kOffsetInfoSize] == 0) {
        continue;
      }
      const auto base = warp * kOffsetInfoSize;
      const auto batch = static_cast<int>(offset_ptr_host[base + 1]);
      const auto table = static_cast<int>(offset_ptr_host[base + 2]);
      const auto indices_start = offset_ptr_host[base + 3];
      const auto indices_end = offset_ptr_host[base + 4];
      const auto num_indices_host = offset_ptr_host[base + 5];
      std::printf(
          "EmbeddingBoundsCheck (VBE %s): (at least one) Out of bounds access "
          "for batch: %d, table: %d, indices_start: %lld, indices_end: %lld, "
          "num_indices: %lld. Setting indices_start and indices_end within "
          "the range.\n",
          vbe ? "true" : "false",
          batch,
          table,
          static_cast<long long>(indices_start),
          static_cast<long long>(indices_end),
          static_cast<long long>(num_indices_host));
      break;
    }

    const int64_t* invalid_ptr_host = invalid_info_host.data_ptr<int64_t>();
    for (int64_t warp = 0; warp < num_warps; ++warp) {
      if (invalid_ptr_host[warp * kInvalidInfoSize] == 0) {
        continue;
      }
      const auto base = warp * kInvalidInfoSize;
      const auto batch = static_cast<int>(invalid_ptr_host[base + 1]);
      const auto table = static_cast<int>(invalid_ptr_host[base + 2]);
      const auto bag_element = invalid_ptr_host[base + 3];
      const auto idx = invalid_ptr_host[base + 4];
      const auto num_rows = invalid_ptr_host[base + 5];
      const auto indices_start = invalid_ptr_host[base + 6];
      const auto indices_end = invalid_ptr_host[base + 7];
      const auto B_local_host = invalid_ptr_host[base + 8];
      const auto b_t = static_cast<int>(invalid_ptr_host[base + 9]);
      std::printf(
          "EmbeddingBoundsCheck (VBE %s): (at least one) Out of bounds access "
          "for batch: %d, table: %d, bag element: %lld, idx: %lld, num_rows: "
          "%lld, indices_start: %lld, indices_end: %lld, T: %d, B: %d, b_t: "
          "%d. Setting idx to zero.\n",
          vbe ? "true" : "false",
          batch,
          table,
          static_cast<long long>(bag_element),
          static_cast<long long>(idx),
          static_cast<long long>(num_rows),
          static_cast<long long>(indices_start),
          static_cast<long long>(indices_end),
          static_cast<int>(T),
          static_cast<int>(B_local_host),
          b_t);
      break;
    }
  }
}
