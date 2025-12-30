/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

#include <ATen/Dispatch.h>

#include <cstdint>
#include <limits>

#ifdef USE_ROCM
#include <rocprim/device/device_radix_sort.hpp>
#else
// clang-format off
#include "fbgemm_gpu/utils/cub_namespace_prefix.cuh"
#include <cub/device/device_radix_sort.cuh>
#include "fbgemm_gpu/utils/cub_namespace_postfix.cuh"
// clang-format on
#endif

using Tensor = at::Tensor;

namespace fbgemm_gpu {

template <
    typename index_t,
    typename scalar_t,
    int UNROLL_FACTOR,
    bool indices_sorted>
__global__ __launch_bounds__(kMaxThreads) void index_select_2d_kernel(
    const pta::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> input,
    const pta::PackedTensorAccessor64<index_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits>
        orig_indices,
    pta::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> output,
    TORCH_DSA_KERNEL_ARGS) {
  const int N = indices.size(0);
  const int input_size = input.size(0);
  const int D = input.size(1);
  CUDA_KERNEL_ASSERT2(output.size(0) == N);

  for (auto row = blockIdx.x; row < N; row += gridDim.x) {
    const index_t src_idx = indices[row];
    const int64_t dst_idx = indices_sorted ? orig_indices[row] : row;
    CUDA_KERNEL_ASSERT2(src_idx < input_size);
    int col;
    for (col = threadIdx.x * UNROLL_FACTOR;
         col < D / UNROLL_FACTOR * UNROLL_FACTOR;
         col += blockDim.x * UNROLL_FACTOR) {
#pragma unroll
      for (int i = 0; i < UNROLL_FACTOR; i++) {
        output[dst_idx][col + i] = LDG(&input[src_idx][col + i]);
      }
    }
    for (; col < D; ++col) {
      output[dst_idx][col] = LDG(&input[src_idx][col]);
    }
  }
}

DLL_PUBLIC Tensor index_select_cuda(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& orig_indices,
    const bool indices_sorted) {
  CUDA_DEVICE_GUARD(input);

  const int N = indices.size(0);
  auto output_shape = input.sizes().vec();
  output_shape[0] = N;

  if (input.numel() == 0 || N == 0) {
    return at::empty(output_shape, input.options());
  }

  Tensor input_reshaped = input.reshape({input.size(0), -1});
  const int D = input_reshaped.size(1);

  Tensor output = at::empty({N, D}, input_reshaped.options());

  const int UNROLL_FACTOR = 2;
  const auto dummy_orig_indices =
      at::empty({0}, at::TensorOptions().dtype(at::kLong));

#define LAUNCH_INDEX_SELECT(INDICES_SORTED)                    \
  {                                                            \
    const auto orig_indices_ =                                 \
        INDICES_SORTED ? orig_indices : dummy_orig_indices;    \
    FBGEMM_LAUNCH_DSA_KERNEL(                                  \
        (index_select_2d_kernel<                               \
            index_t,                                           \
            scalar_t,                                          \
            UNROLL_FACTOR,                                     \
            INDICES_SORTED>),                                  \
        cuda_calc_xblock_count(N, 1),                          \
        std::min(div_round_up(D, UNROLL_FACTOR), kMaxThreads), \
        0,                                                     \
        at::cuda::getCurrentCUDAStream(),                      \
        PTA_B(input_reshaped, scalar_t, 2, 64),                \
        PTA_B(indices, index_t, 1, 64),                        \
        PTA_B(orig_indices_, int64_t, 1, 64),                  \
        PTA_B(output, scalar_t, 2, 64));                       \
  }

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "index_add_2d_kernel_1", [&] {
    FBGEMM_DISPATCH_FLOAT_AND_HALF(
        input_reshaped.scalar_type(), "index_add_2d_kernel_2", [&] {
          if (indices_sorted) {
            LAUNCH_INDEX_SELECT(true)
          } else {
            LAUNCH_INDEX_SELECT(false)
          }
        });
  });

#undef LAUNCH_INDEX_SELECT

  return output.reshape(output_shape);
}

DLL_PUBLIC std::tuple<Tensor, Tensor> sort_indices_with_rocprim(
    const Tensor& indices) {
    TORCH_CHECK(
      indices.dim() == 1,
      "sort_indices_with_rocprim expects a 1D tensor, got ",
      indices.dim());
    TORCH_CHECK(
      indices.is_cuda(),
      "sort_indices_with_rocprim expects a CUDA tensor for indices");

    CUDA_DEVICE_GUARD(indices);
    auto contiguous_indices = indices.contiguous();
    auto sorted_indices = at::empty_like(contiguous_indices);
    auto reverse_indices = at::empty(
      contiguous_indices.sizes(),
      contiguous_indices.options().dtype(at::kLong));
    auto original_positions = at::arange(
      contiguous_indices.numel(),
      contiguous_indices.options().dtype(at::kLong));

    const auto numel = contiguous_indices.numel();
    if (numel == 0) {
      return {sorted_indices, reverse_indices};
    }

    TORCH_CHECK(
      numel <= static_cast<int64_t>(std::numeric_limits<int>::max()),
      "sort_indices_with_rocprim only supports up to INT_MAX elements");

    const int num_items = static_cast<int>(numel);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto scalar_type = contiguous_indices.scalar_type();
    auto dispatch = [&](auto index_value_placeholder) {
      using index_t = decltype(index_value_placeholder);
      auto keys_in = contiguous_indices.data_ptr<index_t>();
      auto keys_out = sorted_indices.data_ptr<index_t>();
      auto values_in = original_positions.data_ptr<int64_t>();
      auto values_out = reverse_indices.data_ptr<int64_t>();

      size_t temp_storage_bytes = 0;

#ifdef USE_ROCM
        // Force ROCm onesweep sort to reduce passes through HBM (ROCm 7.x docs).
        // using sort_config = rocprim::radix_sort_onesweep_config<
        //   rocprim::default_config,
        //   rocprim::default_config,
        //   rocprim::default_config,
        //   400000>;
        // using sort_config = rocprim::radix_sort_onesweep_config<>;
      // Force ROCm onesweep sort to reduce passes through HBM (ROCm 7.x docs).
      // using histogram_cfg = rocprim::kernel_config<256, 12>;
      // using block_cfg = rocprim::kernel_config<256, 12>;
      // constexpr unsinged int kRadixBits = 4;
      // using sort_config = rocprim::radix_sort_onesweep_config<
      //     histogram_cfg,
      //     block_cfg,
      //     RadixBits,
      //     rocprim::block_radix_rank_algorithm::default_algorithm>;
      using sort_config = rocprim::radix_sort_config<
            rocprim::default_config,
            rocprim::default_config,
            rocprim::default_config,
            // 400000>;
            1>;
        AT_CUDA_CHECK(rocprim::radix_sort_pairs<sort_config>(
          nullptr,
          temp_storage_bytes,
          keys_in,
          keys_out,
          values_in,
          values_out,
          num_items,
          0,
          sizeof(index_t) * 8,
          stream,
          false));
      auto temp_storage = at::empty(
          {static_cast<int64_t>(temp_storage_bytes)},
          contiguous_indices.options().dtype(at::kByte));
      AT_CUDA_CHECK(rocprim::radix_sort_pairs<sort_config>(
          temp_storage.data_ptr(),
          temp_storage_bytes,
          keys_in,
          keys_out,
          values_in,
          values_out,
          num_items,
          0,
          sizeof(index_t) * 8,
          stream,
          false));
#else
      AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
          nullptr,
          temp_storage_bytes,
          keys_in,
          keys_out,
          values_in,
          values_out,
          num_items,
          0,
          sizeof(index_t) * 8,
          stream));
      auto temp_storage = at::empty(
          {static_cast<int64_t>(temp_storage_bytes)},
          contiguous_indices.options().dtype(at::kByte));
      AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
          temp_storage.data_ptr(),
          temp_storage_bytes,
          keys_in,
          keys_out,
          values_in,
          values_out,
          num_items,
          0,
          sizeof(index_t) * 8,
          stream));
#endif
    };

    switch (scalar_type) {
      case at::ScalarType::Byte:
        dispatch(uint8_t{});
        break;
      case at::ScalarType::Char:
        dispatch(int8_t{});
        break;
      case at::ScalarType::Short:
        dispatch(int16_t{});
        break;
      case at::ScalarType::Int:
        dispatch(int32_t{});
        break;
      case at::ScalarType::Long:
        dispatch(int64_t{});
        break;
      default:
        TORCH_CHECK(
            false,
            "sort_indices_with_rocprim only supports integral index dtypes, got ",
            scalar_type);
    }

    return {sorted_indices, reverse_indices};
  }

} // namespace fbgemm_gpu
