/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ATen/ops/tensor.h"
#include "c10/core/SymInt.h"
#include "c10/core/TensorOptions.h"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/utils/ops_utils.h"
#include "fbgemm_gpu/utils/tensor_utils.h"
#ifdef USE_ROCM
#include "fbgemm_gpu/utils/rocm/sparse_group_utils.h"
#endif

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <torch/script.h>
#include <cstdint>
#include <stdexcept> // for logic_error
#include <utility>

using Tensor = at::Tensor;

namespace fbgemm_gpu {

namespace {

constexpr int32_t NUM_ARGS = 7;
enum args_pos {
  P_input_ptrs = 0,
  P_output_ptrs = 1,
  P_indices_ptrs = 2,
  P_sorted_indices_ptrs = 3,
  P_reverse_indices_ptrs = 4,
  P_warp_offsets_group_ptrs = 5,
  P_num_cols_group_ptrs = 6,
};

template <typename T>
int64_t compute_num_int64s(const int64_t num_elements) {
  const int64_t ratio = sizeof(int64_t) / sizeof(T);
  return (num_elements + ratio - 1) / ratio;
}

// Compute offsets to set raw pointers
void offset_args(
    int64_t** input_ptrs,
    int64_t** output_ptrs,
    int64_t** indices_ptrs,
    int64_t** sorted_indices_ptrs,
    int64_t** reverse_indices_ptrs,
    int64_t** warp_offsets_group,
    int32_t** num_cols_group,
    int64_t* base_addr,
    const int64_t* const ptr_offsets) {
  *input_ptrs = base_addr + ptr_offsets[P_input_ptrs];
  *output_ptrs = base_addr + ptr_offsets[P_output_ptrs];
  *indices_ptrs = base_addr + ptr_offsets[P_indices_ptrs];
  *sorted_indices_ptrs = base_addr + ptr_offsets[P_sorted_indices_ptrs];
  *reverse_indices_ptrs = base_addr + ptr_offsets[P_reverse_indices_ptrs];
  *warp_offsets_group = base_addr + ptr_offsets[P_warp_offsets_group_ptrs];
  *num_cols_group = reinterpret_cast<int32_t*>(
      base_addr + ptr_offsets[P_num_cols_group_ptrs]);
}

// Struct to hold per-bucket args for split kernel launches
struct SplitArgs {
  int64_t* input_ptrs = nullptr;
  int64_t* output_ptrs = nullptr;
  int64_t* indices_ptrs = nullptr;
  int64_t* sorted_indices_ptrs = nullptr;
  int64_t* reverse_indices_ptrs = nullptr;
  int64_t* warp_offsets_group = nullptr;
  int32_t* num_cols_group = nullptr;
  int64_t total_warps = 0;
  int64_t count = 0;
};

} // namespace

class LookupFunctionBatchedUnaryEmbeddingOp
    : public torch::autograd::Function<LookupFunctionBatchedUnaryEmbeddingOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& weight,
      const Tensor& table_offsets,
      const Tensor& offsets,
      const Tensor& indices) {
    ctx->save_for_backward({weight, table_offsets, offsets, indices});
    auto output = batched_unary_embeddings_forward_cuda(
        weight, table_offsets, offsets, indices);
    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    auto weight = *savedItr++;
    auto table_offsets = *savedItr++;
    auto offsets = *savedItr++;
    auto indices = *savedItr++;
    TORCH_CHECK_VALUE(
        grad_outputs.size() == 1,
        "Expected grad outputs size to be 1, but got ",
        grad_outputs.size());
    // .contiguous() is called on the gradient inputs because
    // the batched_unary_embeddings_backward_cuda assumes contiguous inputs.
    // may cause illegal memory access when it is not
    auto grad_output = grad_outputs[0];
    if (reinterpret_cast<uint64_t>(grad_output.const_data_ptr()) % 16 != 0 ||
        grad_output.stride(1) != 1 || grad_output.stride(0) % 4 != 0) {
      grad_output = grad_output.contiguous();
    }
    if (reinterpret_cast<uint64_t>(grad_output.const_data_ptr()) % 16 != 0) {
      grad_output = at::empty_like(grad_output).copy_(grad_output);
    }
    auto grad_weight = batched_unary_embeddings_backward_cuda(
        grad_output, weight, table_offsets, offsets, indices);
    return {grad_weight, Tensor(), Tensor(), Tensor()};
  }
};

Tensor lookup_batched_unary_embedding_function(
    const Tensor& weight,
    const Tensor& table_offsets,
    const Tensor& offsets,
    const Tensor& indices) {
  return LookupFunctionBatchedUnaryEmbeddingOp::apply(
      weight, table_offsets, offsets, indices)[0];
}

class IndexSelectDim0GPUOp
    : public torch::autograd::Function<IndexSelectDim0GPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& input,
      const Tensor& indices,
      const int consecutive_range_start,
      const int consecutive_range_length,
      const bool skip_indices_sorting_fwd) {
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(input, indices);
    // Expect a 1D index tensor
    TORCH_CHECK_VALUE(
        indices.dim() == 1, "Index tensor must be 1D, but got ", indices.dim());

    Tensor sorted_indices, orig_indices;
    if (skip_indices_sorting_fwd) {
      ctx->save_for_backward({indices});
    } else {
      // Sort indices to promote locality
      std::tie(sorted_indices, orig_indices) = indices.sort();
      ctx->save_for_backward({sorted_indices, orig_indices});
    }

    ctx->saved_data["input_shape"] = input.sizes();
    ctx->saved_data["consecutive_range_start"] = consecutive_range_start;
    ctx->saved_data["consecutive_range_length"] = consecutive_range_length;
    ctx->saved_data["skip_indices_sorting_fwd"] = skip_indices_sorting_fwd;

    return {index_select_cuda(
        input,
        skip_indices_sorting_fwd ? indices : sorted_indices,
        orig_indices,
        /*indices_sorted = */ !skip_indices_sorting_fwd)};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    TORCH_CHECK_VALUE(
        grad_outputs.size() == 1,
        "The size of grad_outputs should be 1, but got ",
        grad_outputs.size());
    TENSOR_ON_CUDA_GPU(grad_outputs[0]);

    bool skip_indices_sorting_fwd =
        ctx->saved_data["skip_indices_sorting_fwd"].toBool();

    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    Tensor sorted_indices;
    Tensor orig_indices;
    if (skip_indices_sorting_fwd) {
      // Sort indices
      Tensor indices = *savedItr++;
      std::tie(sorted_indices, orig_indices) = indices.sort();
    } else {
      sorted_indices = *savedItr++;
      orig_indices = *savedItr++;
    }
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(sorted_indices, orig_indices);
    const Tensor& grad_output = grad_outputs[0];
    TENSORS_ON_SAME_DEVICE(grad_output, sorted_indices);
    auto input_shape = ctx->saved_data["input_shape"].toIntVector();
    int consecutive_range_start =
        ctx->saved_data["consecutive_range_start"].toInt();
    int consecutive_range_length =
        ctx->saved_data["consecutive_range_length"].toInt();

    Tensor undef;
    return {
        index_add_with_unique_indices_cuda(
            grad_output,
            sorted_indices,
            orig_indices,
            input_shape,
            consecutive_range_start,
            consecutive_range_length),
        torch::autograd::Variable(), // indices
        undef, // consecutive_range_start
        undef, // consecutive_range_length
        undef, // skip_indices_sorting_fwd
    };
  }
};

// need to combine input_group and indices_group into one tensor list
// to get this working with autograd.
static torch::autograd::variable_list group_index_select_dim0_forward_impl_gpu(
    at::TensorList all_indices_input,
    const int64_t group_size) {
  // Unpack from TensorList
  auto [input_group, indices_group] =
      group_index_select_dim0_unpack(all_indices_input, group_size);

  // args_tensor stores kernel arguments:
  //   input_ptrs (group_size int64_t elements)
  //   output_ptrs (group_size int64_t elements)
  //   indices_ptrs (group_size int64_t elements)
  //   sorted_indices_ptrs (group_size int64_t elements)
  //   reverse_indices_ptrs (group_size int64_t elements)
  //   warp_offsets_group (group_size + 1 int64_t elements)
  //   num_cols_group (group_size int32_t elements)
  int64_t args_ptrs_offsets[NUM_ARGS + 1];

  const int64_t numels_num_cols_group_64 =
      compute_num_int64s<int32_t>(group_size);

  // Initialize offsets
  args_ptrs_offsets[P_input_ptrs] = group_size;
  args_ptrs_offsets[P_output_ptrs] = group_size;
  args_ptrs_offsets[P_indices_ptrs] = group_size;
  args_ptrs_offsets[P_sorted_indices_ptrs] = group_size;
  args_ptrs_offsets[P_reverse_indices_ptrs] = group_size;
  args_ptrs_offsets[P_warp_offsets_group_ptrs] = group_size + 1;
  args_ptrs_offsets[P_num_cols_group_ptrs] = numels_num_cols_group_64;

  // Compute offsets
  int64_t offset = 0;
  auto next = args_ptrs_offsets[0];
  for (const auto i : c10::irange(NUM_ARGS)) {
    args_ptrs_offsets[i] = offset;
    offset += next;
    next = args_ptrs_offsets[i + 1];
  }
  // Total number of int64_t elements required
  args_ptrs_offsets[NUM_ARGS] = offset;

  // Allocate memory for GroupIndexSelectArgs (split into small/large buckets)
  at::Tensor args_tensor_small = at::empty(
      {static_cast<long>(args_ptrs_offsets[NUM_ARGS] * sizeof(int64_t))},
      at::TensorOptions().dtype(at::kByte).pinned_memory(true));
  at::Tensor args_tensor_large = at::empty(
      {static_cast<long>(args_ptrs_offsets[NUM_ARGS] * sizeof(int64_t))},
      at::TensorOptions().dtype(at::kByte).pinned_memory(true));

  TORCH_CHECK(args_tensor_small.is_contiguous());
  TORCH_CHECK(args_tensor_large.is_contiguous());

  SplitArgs small, large;

  offset_args(
      &small.input_ptrs, &small.output_ptrs, &small.indices_ptrs,
      &small.sorted_indices_ptrs, &small.reverse_indices_ptrs,
      &small.warp_offsets_group, &small.num_cols_group,
      reinterpret_cast<int64_t*>(args_tensor_small.mutable_data_ptr()),
      args_ptrs_offsets);

  offset_args(
      &large.input_ptrs, &large.output_ptrs, &large.indices_ptrs,
      &large.sorted_indices_ptrs, &large.reverse_indices_ptrs,
      &large.warp_offsets_group, &large.num_cols_group,
      reinterpret_cast<int64_t*>(args_tensor_large.mutable_data_ptr()),
      args_ptrs_offsets);

  auto& first_input = input_group[0];
  auto& first_indices = indices_group[0];

  const int input_dim = first_input.dim();
  const int num_output_rows = first_indices.size(0);
  const int num_input_rows = first_input.size(0);
  Tensor input_reshaped = first_input.reshape({num_input_rows, -1});
  const int num_cols = input_reshaped.size(1);
  const int cols_per_warp = get_group_index_select_cols_per_warp();
  const int unroll_factor = get_group_index_select_unroll_factor();

  bool use_var_cols_small = false;
  bool use_var_cols_large = false;
  bool first_small_table = true;
  bool first_large_table = true;
  int prev_num_cols_small = 0;
  int prev_num_cols_large = 0;

  // Allocate memory for output_group
  std::vector<Tensor> output_group;
  output_group.reserve(group_size + 4);

  // We need to store contiguous inputs and indices outside the for-loop to
  // guarantee that the contiguous tensors will outlive the kernel
  // computation
  std::vector<c10::MaybeOwned<at::Tensor>> input_contigs;
  std::vector<c10::MaybeOwned<at::Tensor>> index_contigs;
  input_contigs.reserve(group_size);
  index_contigs.reserve(group_size);

  size_t num_total_indices = 0;

  // For each group, classify into small or large bucket
  for (const auto i : c10::irange(group_size)) {
    const auto& input = input_group[i];
    const auto& indices = indices_group[i];

    // Verify that all input tensors have the same dtype
    TORCH_CHECK_VALUE(
        input.dtype() == first_input.dtype(),
        "All inputs in ",
        group_size,
        " groups need to have the same dtype. Expect group ",
        i,
        ", to be ",
        first_input.dtype(),
        " but got ",
        input.dtype());

    // Verify that all indices have the same dtype
    TORCH_CHECK_VALUE(
        indices.dtype() == first_indices.dtype(),
        "All indices in ",
        group_size,
        " groups need to have the same dtype. Expect group ",
        i,
        ", to be ",
        first_indices.dtype(),
        " but got ",
        indices.dtype());

    // Verify that all input tensors have the same number of dimensions
    TORCH_CHECK_VALUE(
        input_dim == input.dim(),
        "All inputs in group_index_select must have the same number of dimensions. Expect ",
        input_dim,
        " but got group ",
        i,
        " with ",
        input.dim(),
        ". Group size is ",
        group_size);

    // Verify that all tensors are on the same GPU
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(input, indices);

    auto num_output_rows_ = indices.size(0);
    num_total_indices += num_output_rows_;

    // Verify that all input tensors have the same shape[0]
    TORCH_CHECK_VALUE(
        num_output_rows == num_output_rows_,
        "The number of indices to be selected must be the same for the entire group of ",
        group_size,
        ". Expect indices size to be ",
        num_output_rows,
        ", but got group ",
        i,
        " with indices size of ",
        num_output_rows_);
    const auto input_reshaped_ = input.reshape({input.size(0), -1});

    // Number of columns can be different
    auto num_cols_ = input_reshaped_.size(1);

    // Compute warps needed for this table
    int64_t warps_needed;
    const bool is_small = (num_cols_ < cols_per_warp && num_cols_ >= unroll_factor);
    if (is_small) {
      // Packed rows: multiple rows per warp
      int rows_per_warp = cols_per_warp / num_cols_;
      warps_needed = (num_output_rows_ + rows_per_warp - 1) / rows_per_warp;
    } else {
      // Standard: one or more warps per row
      int warps_per_row = (num_cols_ + cols_per_warp - 1) / cols_per_warp;
      warps_needed = warps_per_row * num_output_rows_;
    }

    // Create output pointers
    auto input_shape = input.sizes().vec();
    input_shape[0] = num_output_rows_;
    Tensor output = at::empty(input_shape, input.options());
    TORCH_CHECK(output.is_contiguous(), "output tensor must be contiguous.");
    output_group.push_back(output);

    // Store input and indices contigs to keep them alive during the kernel
    // computation
    input_contigs.push_back(input.expect_contiguous());
    index_contigs.push_back(indices.expect_contiguous());

    // Partition into small or large bucket
    SplitArgs& bucket = is_small ? small : large;
    bool& use_var_cols = is_small ? use_var_cols_small : use_var_cols_large;
    bool& first_table = is_small ? first_small_table : first_large_table;
    int& prev_num_cols = is_small ? prev_num_cols_small : prev_num_cols_large;

    if (!first_table && num_cols_ != prev_num_cols) {
      use_var_cols = true;
    }
    first_table = false;
    prev_num_cols = num_cols_;

    bucket.input_ptrs[bucket.count] =
        reinterpret_cast<int64_t>(input_contigs[i]->const_data_ptr());
    bucket.output_ptrs[bucket.count] =
        reinterpret_cast<int64_t>(output.mutable_data_ptr());
    bucket.indices_ptrs[bucket.count] =
        reinterpret_cast<int64_t>(index_contigs[i]->const_data_ptr());
    bucket.num_cols_group[bucket.count] = num_cols_;
    bucket.warp_offsets_group[bucket.count] = bucket.total_warps;
    bucket.total_warps += warps_needed;
    bucket.count++;
  }

#ifdef USE_ROCM
  // The value is selected empirically. Potential place for optimization.
  constexpr size_t kSortIndicesThreshold = 15'000'000;
  const bool use_sorted_indices_for_bwd =
      (num_total_indices < kSortIndicesThreshold);
#else
  const bool use_sorted_indices_for_bwd = false;
  (void)num_total_indices;
#endif

  // Transfer args tensors to GPU and launch kernels per bucket
  auto transfer_and_offset = [&](at::Tensor& args_tensor, SplitArgs& bucket) {
    bucket.warp_offsets_group[bucket.count] = bucket.total_warps;
    args_tensor = args_tensor.to(first_input.device(), /*non_blocking=*/true);
    offset_args(
        &bucket.input_ptrs, &bucket.output_ptrs, &bucket.indices_ptrs,
        &bucket.sorted_indices_ptrs, &bucket.reverse_indices_ptrs,
        &bucket.warp_offsets_group, &bucket.num_cols_group,
        reinterpret_cast<int64_t*>(args_tensor.mutable_data_ptr()),
        args_ptrs_offsets);
  };

  if (small.count > 0) {
    transfer_and_offset(args_tensor_small, small);
    group_index_select_or_add_cuda(
        small.input_ptrs, small.output_ptrs, small.indices_ptrs,
        /*sorted_indices_ptrs=*/nullptr, /*reverse_indices_ptrs=*/nullptr,
        small.warp_offsets_group, small.num_cols_group,
        first_input.scalar_type(), first_indices.scalar_type(),
        first_input.device().index(), num_output_rows,
        /*total_num_warps=*/small.total_warps, small.count,
        /*use_index_select=*/true, use_var_cols_small,
        /*use_contiguous_warps=*/false, /*use_cache=*/false,
        /*use_packed_rows=*/true);
  }

  if (large.count > 0) {
    transfer_and_offset(args_tensor_large, large);
    group_index_select_or_add_cuda(
        large.input_ptrs, large.output_ptrs, large.indices_ptrs,
        /*sorted_indices_ptrs=*/nullptr, /*reverse_indices_ptrs=*/nullptr,
        large.warp_offsets_group, large.num_cols_group,
        first_input.scalar_type(), first_indices.scalar_type(),
        first_input.device().index(), num_output_rows,
        /*total_num_warps=*/large.total_warps, large.count,
        /*use_index_select=*/true, use_var_cols_large,
        /*use_contiguous_warps=*/false, /*use_cache=*/false,
        /*use_packed_rows=*/false);
  }

  // Build saved_data for each bucket (7 elements each)
  auto make_saved_data = [](int64_t count, bool var_cols, bool packed_rows,
                            bool sorted_indices, int64_t* warp_offsets,
                            int32_t* num_cols, int64_t total_warps) {
    int64_t data[] = {
        count,
        static_cast<int64_t>(var_cols),
        static_cast<int64_t>(packed_rows),
        static_cast<int64_t>(sorted_indices),
        reinterpret_cast<int64_t>(warp_offsets),
        reinterpret_cast<int64_t>(num_cols),
        total_warps,
    };
    auto t = at::empty(
        {sizeof(data) / sizeof(int64_t)}, at::TensorOptions().dtype(at::kLong));
    TORCH_CHECK(t.is_contiguous());
    memcpy(t.mutable_data_ptr<int64_t>(), data, sizeof(data));
    return t;
  };

  auto saved_data_t_small = make_saved_data(
      small.count, use_var_cols_small, /*packed_rows=*/true,
      use_sorted_indices_for_bwd, small.warp_offsets_group,
      small.num_cols_group, small.total_warps);
  auto saved_data_t_large = make_saved_data(
      large.count, use_var_cols_large, /*packed_rows=*/false,
      use_sorted_indices_for_bwd, large.warp_offsets_group,
      large.num_cols_group, large.total_warps);

  output_group.push_back(args_tensor_small);
  output_group.push_back(args_tensor_large);
  output_group.push_back(saved_data_t_small);
  output_group.push_back(saved_data_t_large);

  // return format:
  // (group_size outputs, 2 args_tensors, 2 saved_data)
  return output_group;
}

static torch::autograd::variable_list group_index_select_dim0_backward_impl_gpu(
    at::TensorList all_inputs,
    c10::SymIntArrayRef output_shape_group_ref) {
  TORCH_CHECK_VALUE(
      all_inputs.size() > 4,
      "all_inputs size must be larger than 4, but got ",
      all_inputs.size());

  // all_input size = group_size * 2 (grads, indices)
  // + 2 args_tensors + 2 saved_data + 1 first input
  const int64_t group_size = (all_inputs.size() - 5) / 2;

  const Tensor& fwd_input = all_inputs[2 * group_size + 4];
  const Tensor& saved_data_small_t = all_inputs[2 * group_size + 2];
  const Tensor& saved_data_large_t = all_inputs[2 * group_size + 3];
  const Tensor& first_indices = all_inputs[group_size];
  const int64_t output_dim = fwd_input.dim();

  auto grad_output_group = std::vector<Tensor>(
      all_inputs.cbegin(), all_inputs.cbegin() + group_size);
  std::vector<int64_t> output_shape_group;
  output_shape_group.reserve(output_shape_group_ref.size());
  for (const auto& i : output_shape_group_ref) {
    output_shape_group.push_back(i.as_int_unchecked());
  }

  auto indices_group = std::vector<Tensor>(
      all_inputs.cbegin() + group_size, all_inputs.cbegin() + 2 * group_size);

  // Helper to unpack saved_data
  struct BucketData {
    int64_t count;
    bool use_var_cols;
    bool use_packed_rows;
    bool use_sorted_indices;
    int64_t* warp_offsets_group;
    int32_t* num_cols_group;
    int64_t total_num_warps;
  };

  auto unpack_saved_data = [](const Tensor& t) -> BucketData {
    TORCH_CHECK(t.device() == at::kCPU);
    TORCH_CHECK(t.is_contiguous());
    const int64_t* p = t.const_data_ptr<int64_t>();
    return {
        p[0],
        static_cast<bool>(p[1]),
        static_cast<bool>(p[2]),
        static_cast<bool>(p[3]),
        reinterpret_cast<int64_t*>(p[4]),
        reinterpret_cast<int32_t*>(p[5]),
        p[6],
    };
  };

  auto sd_small = unpack_saved_data(saved_data_small_t);
  auto sd_large = unpack_saved_data(saved_data_large_t);

  TORCH_CHECK_VALUE(
      (sd_small.count + sd_large.count) == group_size,
      "count_small + count_large must match group_size. Expect ",
      group_size, " but got ", (sd_small.count + sd_large.count));

  // We checked in forward that all output rows are the same for all member
  // in the group
  const int num_input_rows = grad_output_group[0].size(0);

  std::vector<Tensor> outputs;
  outputs.reserve(group_size * 2 + 1);

  // 1) Add group_size placeholder Variable()'s for indices
  {
    const auto placeholder =
        at::empty({0}, at::TensorOptions().dtype(at::kLong));
    for (auto i = 0; i < group_size; i++) {
      outputs.push_back(placeholder);
    }
  }

  // Allocate backward args tensors for each bucket (5 ptrs per group member)
  Tensor args_tensor_small = at::empty(
      {sd_small.count * 5},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  Tensor args_tensor_large = at::empty(
      {sd_large.count * 5},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  TORCH_CHECK(args_tensor_small.is_contiguous());
  TORCH_CHECK(args_tensor_large.is_contiguous());

  int64_t* grad_output_ptrs_small = args_tensor_small.mutable_data_ptr<int64_t>();
  int64_t* grad_input_ptrs_small = grad_output_ptrs_small + sd_small.count;
  int64_t* indices_ptrs_small = grad_input_ptrs_small + sd_small.count;
  int64_t* sorted_indices_ptrs_small = indices_ptrs_small + sd_small.count;
  int64_t* reverse_indices_ptrs_small = sorted_indices_ptrs_small + sd_small.count;

  int64_t* grad_output_ptrs_large = args_tensor_large.mutable_data_ptr<int64_t>();
  int64_t* grad_input_ptrs_large = grad_output_ptrs_large + sd_large.count;
  int64_t* indices_ptrs_large = grad_input_ptrs_large + sd_large.count;
  int64_t* sorted_indices_ptrs_large = indices_ptrs_large + sd_large.count;
  int64_t* reverse_indices_ptrs_large = sorted_indices_ptrs_large + sd_large.count;

  const int cols_per_warp = get_group_index_select_cols_per_warp();

  int64_t group_grad_input_numel = 0;
  std::vector<int64_t> grad_input_numels;
  grad_input_numels.reserve(group_size);

  std::vector<c10::MaybeOwned<at::Tensor>> grad_output_contigs;
  grad_output_contigs.reserve(group_size);

  int64_t idx_small = 0;
  int64_t idx_large = 0;

  for (const auto i : c10::irange(group_size)) {
    const auto& grad = grad_output_group[i];
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(grad, first_indices);

    grad_output_contigs.push_back(grad.expect_contiguous());

    int64_t grad_input_numel = output_shape_group[i * output_dim];
    for (auto j = (i * output_dim) + 1; j < (i + 1) * output_dim; j++) {
      grad_input_numel *= output_shape_group[j];
    }
    grad_input_numels.push_back(grad_input_numel);
    group_grad_input_numel += grad_input_numel;

    // Partition grad_output_ptrs into small/large
    const auto grad_reshaped = grad.reshape({grad.size(0), -1});
    const auto num_cols_ = grad_reshaped.size(1);
    if (num_cols_ < cols_per_warp) {
      grad_output_ptrs_small[idx_small++] =
          reinterpret_cast<int64_t>(grad_output_contigs[i]->const_data_ptr());
    } else {
      grad_output_ptrs_large[idx_large++] =
          reinterpret_cast<int64_t>(grad_output_contigs[i]->const_data_ptr());
    }
  }

  // Allocate a big tensor to avoid calling many small elementwise kernels
  const auto group_grad_input =
      at::zeros({group_grad_input_numel}, fwd_input.options());
  TORCH_CHECK(group_grad_input.is_contiguous());

  auto output_group = group_grad_input.split(grad_input_numels, 0);
  TORCH_CHECK_VALUE(
      output_group.size() == static_cast<size_t>(group_size),
      "output_group size must be ", group_size,
      " but got ", output_group.size());

  // Reshape grad inputs and partition pointers
  idx_small = 0;
  idx_large = 0;
  for (int i = 0; i < group_size; i++) {
    output_group[i] = output_group[i].reshape(c10::IntArrayRef(
        output_shape_group.data() + i * output_dim, output_dim));
    TORCH_CHECK(output_group[i].is_contiguous());

    const auto reshaped = grad_output_group[i].reshape(
        {grad_output_group[i].size(0), -1});
    const auto num_cols_ = reshaped.size(1);
    if (num_cols_ < cols_per_warp) {
      grad_input_ptrs_small[idx_small++] =
          reinterpret_cast<int64_t>(output_group[i].const_data_ptr());
    } else {
      grad_input_ptrs_large[idx_large++] =
          reinterpret_cast<int64_t>(output_group[i].const_data_ptr());
    }

    outputs.push_back(output_group[i]);
  }

  // Calculate indices_ptrs and partition into buckets
  std::vector<c10::MaybeOwned<at::Tensor>> index_contigs;
  index_contigs.reserve(group_size);
  // Keep per-bucket index tensor lists for sorting
  std::vector<at::Tensor> index_tensors_small;
  std::vector<at::Tensor> index_tensors_large;
  index_tensors_small.reserve(sd_small.count);
  index_tensors_large.reserve(sd_large.count);
  idx_small = 0;
  idx_large = 0;
  for (const auto i : c10::irange(group_size)) {
    const auto& indices = indices_group[i];
    index_contigs.push_back(indices.expect_contiguous());

    const auto reshaped = grad_output_group[i].reshape(
        {grad_output_group[i].size(0), -1});
    const auto num_cols_ = reshaped.size(1);
    if (num_cols_ < cols_per_warp) {
      indices_ptrs_small[idx_small] =
          reinterpret_cast<int64_t>(index_contigs[i]->const_data_ptr());
      sorted_indices_ptrs_small[idx_small] = 0;
      reverse_indices_ptrs_small[idx_small] = 0;
      index_tensors_small.push_back(*index_contigs[i]);
      idx_small++;
    } else {
      indices_ptrs_large[idx_large] =
          reinterpret_cast<int64_t>(index_contigs[i]->const_data_ptr());
      sorted_indices_ptrs_large[idx_large] = 0;
      reverse_indices_ptrs_large[idx_large] = 0;
      index_tensors_large.push_back(*index_contigs[i]);
      idx_large++;
    }
  }

#ifdef USE_ROCM
  // Per-bucket sorting
  const int64_t sort_num_items = static_cast<int64_t>(indices_group[0].numel());
  const bool use_segmented_sort =
      static_cast<size_t>(sort_num_items) <= rocm::k_sort_merge_threshold;

  // Sorting helper for a single bucket. Returns the sort output tensors
  // so they stay alive until after the kernel launch.
  auto sort_bucket = [&](int64_t count, bool use_sorted,
                         int64_t* bucket_indices_ptrs,
                         int64_t* bucket_sorted_ptrs,
                         int64_t* bucket_reverse_ptrs,
                         const std::vector<at::Tensor>& bucket_index_tensors)
      -> std::pair<at::Tensor, at::Tensor> {
    if (!use_sorted || count == 0) return {at::Tensor(), at::Tensor()};

    const auto stream = at::cuda::getCurrentCUDAStream();
    const size_t temp_bytes = use_segmented_sort
        ? rocm::get_segmented_sort_temp_storage_bytes(
              static_cast<size_t>(sort_num_items), count,
              first_indices.scalar_type(), stream)
        : rocm::get_sort_temp_storage_bytes(
              static_cast<size_t>(sort_num_items),
              first_indices.scalar_type(), stream);
    auto sort_temp = at::empty(
        {static_cast<int64_t>(temp_bytes)},
        first_indices.options().dtype(at::kByte));
    auto sort_positions = at::arange(
        sort_num_items, first_indices.options().dtype(at::kLong));
    auto sorted_indices = at::empty(
        {count * sort_num_items}, first_indices.options());
    auto reverse_indices = at::empty(
        {count * sort_num_items},
        first_indices.options().dtype(at::kLong));
    sorted_indices.record_stream(stream);
    reverse_indices.record_stream(stream);

    // Fill sorted/reverse ptr tables
    const int64_t idx_elem_bytes = first_indices.element_size();
    auto* sorted_base = static_cast<char*>(sorted_indices.data_ptr());
    auto* reverse_base = static_cast<char*>(reverse_indices.data_ptr());
    for (int64_t j = 0; j < count; ++j) {
      bucket_sorted_ptrs[j] = reinterpret_cast<int64_t>(
          sorted_base + j * sort_num_items * idx_elem_bytes);
      bucket_reverse_ptrs[j] = reinterpret_cast<int64_t>(
          reverse_base + j * sort_num_items * sizeof(int64_t));
    }

    if (use_segmented_sort) {
      const auto all_keys_in = at::cat(bucket_index_tensors, 0);
      const auto all_values_in =
          sort_positions.unsqueeze(0)
              .expand({count, sort_num_items})
              .contiguous()
              .view({-1});
      const auto segment_offsets =
          at::arange(count + 1, first_indices.options().dtype(at::kLong)) *
          sort_num_items;
      rocm::sort_indices_segmented_rocprim(
          all_keys_in, sorted_indices, all_values_in, reverse_indices,
          segment_offsets, static_cast<size_t>(sort_num_items), count,
          sort_temp, stream);
    } else {
      rocm::sort_indices_batch_rocprim(
          bucket_indices_ptrs, sorted_indices.data_ptr(),
          reverse_indices.data_ptr<int64_t>(),
          sort_positions.data_ptr<int64_t>(),
          static_cast<size_t>(sort_num_items), count,
          sort_temp, first_indices.scalar_type(), stream);
    }
    return {sorted_indices, reverse_indices};
  };

  // Keep sort output tensors alive until after kernel launches
  auto [sorted_small, reverse_small] = sort_bucket(
      sd_small.count, sd_small.use_sorted_indices,
      indices_ptrs_small, sorted_indices_ptrs_small,
      reverse_indices_ptrs_small, index_tensors_small);
  auto [sorted_large, reverse_large] = sort_bucket(
      sd_large.count, sd_large.use_sorted_indices,
      indices_ptrs_large, sorted_indices_ptrs_large,
      reverse_indices_ptrs_large, index_tensors_large);
#endif

  // Transfer backward args tensors to GPU and launch kernels
  args_tensor_small = args_tensor_small.to(
      first_indices.device(), /*non_blocking=*/true);
  args_tensor_large = args_tensor_large.to(
      first_indices.device(), /*non_blocking=*/true);

  auto launch_backward = [&](const BucketData& sd, Tensor& args_tensor,
                             bool use_packed_rows) {
    if (sd.count == 0) return;
    const bool use_sorted = sd.use_sorted_indices;
    group_index_select_or_add_cuda(
        args_tensor.const_data_ptr<int64_t>(),
        args_tensor.const_data_ptr<int64_t>() + sd.count,
        args_tensor.const_data_ptr<int64_t>() + 2 * sd.count,
        use_sorted ? args_tensor.const_data_ptr<int64_t>() + 3 * sd.count : nullptr,
        use_sorted ? args_tensor.const_data_ptr<int64_t>() + 4 * sd.count : nullptr,
        sd.warp_offsets_group, sd.num_cols_group,
        fwd_input.scalar_type(), first_indices.scalar_type(),
        fwd_input.device().index(), num_input_rows,
        sd.total_num_warps, sd.count,
        /*use_index_select=*/false, sd.use_var_cols,
        /*use_contiguous_warps=*/use_sorted,
        /*use_cache=*/use_sorted,
        use_packed_rows);
  };

  launch_backward(sd_small, args_tensor_small, /*use_packed_rows=*/true);
  launch_backward(sd_large, args_tensor_large, /*use_packed_rows=*/false);

  return outputs;
}

Tensor pack_segments_cuda(
    const Tensor& t_in,
    const Tensor& lengths,
    const int64_t max_length) {
  return fbgemm_gpu::pack_segments_forward_cuda(t_in, lengths, max_length)[0];
}

std::tuple<Tensor, std::optional<Tensor>> pack_segments_cuda_v2(
    const Tensor& t_in,
    const Tensor& lengths,
    const int64_t max_length,
    const bool pad_minf,
    const bool return_presence_mask) {
  return fbgemm_gpu::pack_segments_forward_cuda_v2(
      t_in, lengths, max_length, pad_minf, return_presence_mask);
}

Tensor index_select_dim0_gpu(
    const Tensor& input,
    const Tensor& indices,
    std::optional<int64_t> consecutive_range_start,
    std::optional<int64_t> consecutive_range_length,
    std::optional<bool> skip_indices_sorting_fwd) {
  bool user_skip_indices_sorting_fwd =
      skip_indices_sorting_fwd ? *skip_indices_sorting_fwd : false;
  return IndexSelectDim0GPUOp::apply(
      input,
      indices,
      consecutive_range_start ? *consecutive_range_start : 0,
      consecutive_range_length ? *consecutive_range_length : 0,
      // Always skip indices sorting if doing forward only
      user_skip_indices_sorting_fwd && !c10::InferenceMode::is_enabled())[0];
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA(
      "reorder_batched_ad_lengths", fbgemm_gpu::reorder_batched_ad_lengths_gpu);
  DISPATCH_TO_CUDA(
      "reorder_batched_ad_indices", fbgemm_gpu::reorder_batched_ad_indices_gpu);
  DISPATCH_TO_CUDA(
      "reorder_batched_sequence_embeddings",
      fbgemm_gpu::reorder_batched_sequence_embeddings_gpu);
  DISPATCH_TO_CUDA(
      "batched_unary_embeddings",
      fbgemm_gpu::lookup_batched_unary_embedding_function);
  DISPATCH_TO_CUDA(
      "histogram_binning_calibration",
      fbgemm_gpu::histogram_binning_calibration_cuda);
  DISPATCH_TO_CUDA(
      "histogram_binning_calibration_by_feature",
      fbgemm_gpu::histogram_binning_calibration_by_feature_cuda);
  DISPATCH_TO_CUDA(
      "generic_histogram_binning_calibration_by_feature",
      fbgemm_gpu::generic_histogram_binning_calibration_by_feature_cuda);
  DISPATCH_TO_CUDA("pack_segments", fbgemm_gpu::pack_segments_forward_cuda);
  DISPATCH_TO_CUDA(
      "pack_segments_v2", fbgemm_gpu::pack_segments_forward_cuda_v2);
  DISPATCH_TO_CUDA(
      "pack_segments_backward", fbgemm_gpu::pack_segments_backward_cuda);
  DISPATCH_TO_CUDA("index_select_dim0", fbgemm_gpu::index_select_dim0_gpu);
  DISPATCH_TO_CUDA(
      "group_index_select_dim0_gpu_impl",
      fbgemm_gpu::group_index_select_dim0_forward_impl_gpu);
  DISPATCH_TO_CUDA(
      "group_index_select_dim0_gpu_backward",
      fbgemm_gpu::group_index_select_dim0_backward_impl_gpu);
  DISPATCH_TO_CUDA(
      "group_index_select_dim0", fbgemm_gpu::group_index_select_dim0);
}

TORCH_LIBRARY_IMPL(fbgemm, AutogradCUDA, m) {
  m.impl("group_index_select_dim0", &fbgemm_gpu::group_index_select_dim0);
  m.impl(
      "group_index_select_dim0_gpu_impl",
      &fbgemm_gpu::group_index_select_dim0_autograd_impl);
}
