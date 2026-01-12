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

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <torch/script.h>
#include <cstdint>
#include <stdexcept> // for logic_error

using Tensor = at::Tensor;

namespace fbgemm_gpu {

namespace {

constexpr int32_t NUM_ARGS = 5;
enum args_pos {
  P_input_ptrs = 0,
  P_output_ptrs = 1,
  P_indices_ptrs = 2,
  P_warp_offsets_group_ptrs = 3,
  P_num_cols_group_ptrs = 4
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
    int64_t** warp_offsets_group,
    int32_t** num_cols_group,
    int64_t* base_addr,
    const int64_t* const ptr_offsets) {
  *input_ptrs = base_addr + ptr_offsets[P_input_ptrs];
  *output_ptrs = base_addr + ptr_offsets[P_output_ptrs];
  *indices_ptrs = base_addr + ptr_offsets[P_indices_ptrs];
  *warp_offsets_group = base_addr + ptr_offsets[P_warp_offsets_group_ptrs];
  *num_cols_group = reinterpret_cast<int32_t*>(
      base_addr + ptr_offsets[P_num_cols_group_ptrs]);
}
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
    TORCH_CHECK(grad_outputs.size() == 1);
    // .contiguous() is called on the gradient inputs because
    // the batched_unary_embeddings_backward_cuda assumes contiguous inputs.
    // may cause illegal memory access when it is not
    auto grad_output = grad_outputs[0];
    if (reinterpret_cast<uint64_t>(grad_output.data_ptr()) % 16 != 0 ||
        grad_output.stride(1) != 1 || grad_output.stride(0) % 4 != 0) {
      grad_output = grad_output.contiguous();
    }
    if (reinterpret_cast<uint64_t>(grad_output.data_ptr()) % 16 != 0) {
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
    TORCH_CHECK(indices.dim() == 1, "Index tensor must be 1D")

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
    TORCH_CHECK(grad_outputs.size() == 1);
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
  //   warp_offsets_group (group_size + 1 int64_t elements)
  //   num_cols_group (group_size int32_t elements)
  int64_t args_ptrs_offsets[NUM_ARGS + 1];

  const int64_t numels_num_cols_group_64 =
      compute_num_int64s<int32_t>(group_size);

  // Initialize offsets
  args_ptrs_offsets[P_input_ptrs] = group_size;
  args_ptrs_offsets[P_output_ptrs] = group_size;
  args_ptrs_offsets[P_indices_ptrs] = group_size;
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

  // Allocate memory for GroupIndexSelectArgs
  at::Tensor args_tensor_small = at::empty(
      {static_cast<long>(args_ptrs_offsets[NUM_ARGS] * sizeof(int64_t))},
      at::TensorOptions().dtype(at::kByte).pinned_memory(true));

  at::Tensor args_tensor_large = at::empty(
      {static_cast<long>(args_ptrs_offsets[NUM_ARGS] * sizeof(int64_t))},
      at::TensorOptions().dtype(at::kByte).pinned_memory(true));

  // Ensure that args_tensors are contiguous
  TORCH_CHECK(args_tensor_small.is_contiguous());
  TORCH_CHECK(args_tensor_large.is_contiguous());

  // defining a struct that will maintain the arguments required by
  // the GPU kernel
  struct SplitArgs {
    int64_t* input_ptrs = nullptr;
    int64_t* output_ptrs = nullptr;
    int64_t* indices_ptrs = nullptr;
    int64_t* warp_offsets_group = nullptr;
    int32_t* num_cols_group = nullptr;
    int64_t total_warps = 0;
    int64_t count = 0;
  } small, large; // small and large structs to hold args for small and large
     // tables respectively

  // Offset host pointers
  offset_args(
      &small.input_ptrs,
      &small.output_ptrs,
      &small.indices_ptrs,
      &small.warp_offsets_group,
      &small.num_cols_group,
      reinterpret_cast<int64_t*>(args_tensor_small.data_ptr()),
      args_ptrs_offsets);

  offset_args(
      &large.input_ptrs,
      &large.output_ptrs,
      &large.indices_ptrs,
      &large.warp_offsets_group,
      &large.num_cols_group,
      reinterpret_cast<int64_t*>(args_tensor_large.data_ptr()),
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
  int64_t warp_offset = 0;
  bool use_var_cols = false;

  // Allocate memory for output_group
  std::vector<Tensor> output_group;
  output_group.reserve(group_size + 2);

  // We need to store contiguous inputs and indices outside the for-loop to
  // guarantee that the contiguous tensors will outlive the kernel
  // computation
  std::vector<c10::MaybeOwned<at::Tensor>> input_contigs;
  std::vector<c10::MaybeOwned<at::Tensor>> index_contigs;
  input_contigs.reserve(group_size);
  index_contigs.reserve(group_size);

  // For each group, copy input to output
  for (const auto i : c10::irange(group_size)) {
    const auto& input = input_group[i];
    const auto& indices = indices_group[i];

    // Verify that all input tensors have the same number of dimensions
    TORCH_CHECK(
        input_dim == input.dim(),
        "All inputs in group_index_select must have the same number of dimensions");

    // Verify that all tensors are on the same GPU
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(input, indices);

    auto num_output_rows_ = indices.size(0);

    // Verify that all input tensors have the same shape[0]
    TORCH_CHECK(
        num_output_rows == num_output_rows_,
        "The number of indices to be selected must be the same for the entire group");
    const auto input_reshaped_ = input.reshape({input.size(0), -1});

    // Number of columns can be different
    auto num_cols_ = input_reshaped_.size(1);

#ifdef USE_ROCM
    int64_t warps_needed;
    if (num_cols_ < cols_per_warp && num_cols_ >= unroll_factor) {
      // Optimization: Pack multiple rows into one warp
      int rows_per_warp = cols_per_warp / num_cols_;
      warps_needed = (num_output_rows_ + rows_per_warp - 1) / rows_per_warp;
    } else {
      // Standard: One or more warps per row
      int warps_per_row = (num_cols_ + cols_per_warp - 1) / cols_per_warp;
      warps_needed = warps_per_row * num_output_rows_;
    }
#else
    // Standard: One or more warps per row
    auto warps_per_row = (num_cols_ + cols_per_warp - 1) / cols_per_warp;
#endif // USE_ROCM

    // TODO: maintain [use_var_cols] separately for small emb dims
    if (num_cols != num_cols_) {
      use_var_cols = true;
    }

    // Create output pointers
    auto input_shape = input.sizes().vec();
    input_shape[0] = num_output_rows_;
    Tensor output = at::empty(input_shape, input.options());
    // Ensure that the allocated output is contiguous
    TORCH_CHECK(output.is_contiguous())
    output_group.push_back(output);

    // Store input and indices contigs to keep them alive during the kernel
    // computation
    input_contigs.push_back(input.expect_contiguous());
    index_contigs.push_back(indices.expect_contiguous());

<<<<<<< HEAD
    if (num_cols_ < cols_per_warp) {
        // Optimization for Small Embedding: Pack multiple rows per warp
        small.input_ptrs[small.count] = reinterpret_cast<int64_t>(input_contigs[i]->data_ptr());
        small.output_ptrs[small.count] = reinterpret_cast<int64_t>(output.data_ptr());
        small.indices_ptrs[small.count] = reinterpret_cast<int64_t>(index_contigs[i]->data_ptr());
        small.num_cols_group[small.count] = num_cols_;
        small.warp_offsets_group[small.count] = small.total_warps;
        small.total_warps += warps_needed;
        small.count++;
    } else {
        // Standard Embedding: One or more warps per row
        large.input_ptrs[large.count] = reinterpret_cast<int64_t>(input_contigs[i]->data_ptr());
        large.output_ptrs[large.count] = reinterpret_cast<int64_t>(output.data_ptr());
        large.indices_ptrs[large.count] = reinterpret_cast<int64_t>(index_contigs[i]->data_ptr());
        large.num_cols_group[large.count] = num_cols_;
        large.warp_offsets_group[large.count] = large.total_warps;
        large.total_warps += warps_needed;
        large.count++;
    }
=======
    // Store args
    input_ptrs[i] = reinterpret_cast<int64_t>(input_contigs[i]->data_ptr());
    output_ptrs[i] = reinterpret_cast<int64_t>(output.data_ptr());
    indices_ptrs[i] = reinterpret_cast<int64_t>(index_contigs[i]->data_ptr());
    warp_offsets_group[i] = warp_offset;
    num_cols_group[i] = num_cols_;

#ifdef USE_ROCM
    warp_offset += warps_needed;
#else
    warp_offset += warps_per_row * num_output_rows;
#endif // USE_ROCM
>>>>>>> aryaman/group-index-subwarp
  }

  // Store the last offset
  if (small.count > 0) {
    small.warp_offsets_group[small.count] = small.total_warps;
  }
  if (large.count > 0) {
    large.warp_offsets_group[large.count] = large.total_warps;
  }

  // Transfer args tensors to GPU
  if(small.count > 0) {
    args_tensor_small = args_tensor_small.to(
        first_input.device(),
        /*non_blocking=*/true);
        
        // Offset raw ptrs in GPU memory
        offset_args(
            &small.input_ptrs,
            &small.output_ptrs,
            &small.indices_ptrs,
            &small.warp_offsets_group,
            &small.num_cols_group,
            reinterpret_cast<int64_t*>(args_tensor_small.data_ptr()),
            args_ptrs_offsets);
  }

  if(large.count > 0) {
    args_tensor_large = args_tensor_large.to(
        first_input.device(),
        /*non_blocking=*/true);
        
        // Offset raw ptrs in GPU memory
        offset_args(
            &large.input_ptrs,
            &large.output_ptrs,
            &large.indices_ptrs,
            &large.warp_offsets_group,
            &large.num_cols_group,
            reinterpret_cast<int64_t*>(args_tensor_large.data_ptr()),
            args_ptrs_offsets);
  }  

  int64_t saved_data_small[] = {
      static_cast<int64_t>(small.count),
      use_var_cols,
      reinterpret_cast<int64_t>(small.warp_offsets_group),
      reinterpret_cast<int64_t>(small.num_cols_group),
      small.total_warps,
  };
  auto saved_data_t_small = at::empty(
      {sizeof(saved_data_small) / sizeof(int64_t)},
      at::TensorOptions().dtype(at::kLong));
  TORCH_CHECK(saved_data_t_small.is_contiguous());
  memcpy(saved_data_t_small.data_ptr<int64_t>(), saved_data_small, sizeof(saved_data_small));

  int64_t saved_data_large[] = {
      static_cast<int64_t>(large.count),
      use_var_cols,
      reinterpret_cast<int64_t>(large.warp_offsets_group),
      reinterpret_cast<int64_t>(large.num_cols_group),
      large.total_warps,
  };
  auto saved_data_t_large = at::empty(
      {sizeof(saved_data_large) / sizeof(int64_t)},
      at::TensorOptions().dtype(at::kLong));
  TORCH_CHECK(saved_data_t_large.is_contiguous());
  memcpy(saved_data_t_large.data_ptr<int64_t>(), saved_data_large, sizeof(saved_data_large));

  if(small.count > 0) {
    group_index_select_or_add_cuda_smallEmbD(
        small.input_ptrs,
        small.output_ptrs,
        small.indices_ptrs,
        small.warp_offsets_group,
        small.num_cols_group,
        first_input.scalar_type(),
        first_indices.scalar_type(),
        first_input.device().index(),
        num_output_rows,
        /*total_num_warps=*/small.total_warps,
        small.count,
        /*use_index_select=*/true,
        use_var_cols);
  }

  if(large.count > 0) {
    group_index_select_or_add_cuda(
        large.input_ptrs,
        large.output_ptrs,
        large.indices_ptrs,
        large.warp_offsets_group,
        large.num_cols_group,
        first_input.scalar_type(),
        first_indices.scalar_type(),
        first_input.device().index(),
        num_output_rows,
        /*total_num_warps=*/large.total_warps,
        large.count,
        /*use_index_select=*/true,
        use_var_cols);
  }

  output_group.push_back(args_tensor_small);
  output_group.push_back(args_tensor_large);
  output_group.push_back(saved_data_t_small);
  output_group.push_back(saved_data_t_large);

  // return format:
  // (group_size outputs, 2 args_tensor2, 2 saved_data)
  return output_group;
}

static torch::autograd::variable_list group_index_select_dim0_backward_impl_gpu(
    at::TensorList all_inputs,
    c10::SymIntArrayRef output_shape_group_ref) {
  TORCH_CHECK(all_inputs.size() > 4);

  // all_input size =  group_size * 2 (from grads, indices)
  // + 2 args_tensor2 + 2 saved_data + 1 first input
  const int64_t group_size = (all_inputs.size() - 5) / 2;

  const Tensor& fwd_input = all_inputs[2 * group_size + 4];
  const Tensor& saved_data_small = all_inputs[2 * group_size + 2];
  const Tensor& saved_data_large = all_inputs[2 * group_size + 3];
  
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

  // Retrieve saved data
  TORCH_CHECK(saved_data_small.device() == at::kCPU);
  TORCH_CHECK(saved_data_small.is_contiguous());
  int64_t* saved_data_small_ptr = saved_data_small.data_ptr<int64_t>();
  auto count_small = saved_data_small_ptr[0];
  const bool use_var_cols = saved_data_small_ptr[1];
  int64_t* warp_offsets_group_small = reinterpret_cast<int64_t*>(saved_data_small_ptr[2]);
  int32_t* num_cols_group_small = reinterpret_cast<int32_t*>(saved_data_small_ptr[3]);
  int64_t total_num_warps_small = saved_data_small_ptr[4];

  TORCH_CHECK(saved_data_large.device() == at::kCPU);
  TORCH_CHECK(saved_data_large.is_contiguous());
  int64_t* saved_data_large_ptr = saved_data_large.data_ptr<int64_t>();
  auto count_large = saved_data_large_ptr[0];
  const bool use_var_cols_large = saved_data_large_ptr[1];
  int64_t* warp_offsets_group_large = reinterpret_cast<int64_t*>(saved_data_large_ptr[2]);
  int32_t* num_cols_group_large = reinterpret_cast<int32_t*>(saved_data_large_ptr[3]);
  int64_t total_num_warps_large = saved_data_large_ptr[4];  

  // We checked in forward that all output rows are the same for all member
  // in the group
  const int num_input_rows = grad_output_group[0].size(0);

  std::vector<Tensor> outputs;
  // Returning 3 outputs:
  // 1) group_size Variable()'s for indices
  // 2) group_size gradients for inputs
  // 3) 1 Variable() for group_size
  outputs.reserve(group_size * 2 + 1);

  // 1) Add group_size Variable()'s for indices
  // c10::irange cannot be used in here as it
  // triggers a build error of i being an unused variable.
  // Add empty tensor with zero size here to make __torch_dispatch__ work for
  // the backward op. Those empty tensors will be replaced with
  // torch::autograd::Variable() outside of the op call.
  for (auto i = 0; i < group_size; i++) {
    outputs.push_back(at::empty({0}, at::TensorOptions().dtype(at::kLong)));
  }

  // Allocate Tensors for ptrs of grad output and input, and indices
  Tensor args_tensor_small = at::empty(
      {count_small * 3},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));

  Tensor args_tensor_large = at::empty(
      {count_large * 3},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  // Ensure that args_tensors are contiguous
  TORCH_CHECK(args_tensor_small.is_contiguous());
  TORCH_CHECK(args_tensor_large.is_contiguous());

  int64_t* grad_output_ptrs_small = args_tensor_small.data_ptr<int64_t>();
  int64_t* grad_input_ptrs_small = args_tensor_small.data_ptr<int64_t>() + count_small;
  int64_t* indices_ptrs_small = args_tensor_small.data_ptr<int64_t>() + 2 * count_small;

  int64_t* grad_output_ptrs_large = args_tensor_large.data_ptr<int64_t>();
  int64_t* grad_input_ptrs_large = args_tensor_large.data_ptr<int64_t>() + count_large;
  int64_t* indices_ptrs_large = args_tensor_large.data_ptr<int64_t>() + 2 * count_large;

  int64_t group_grad_input_numel = 0;
  std::vector<int64_t> grad_input_numels;
  grad_input_numels.reserve(group_size);

  // We need to store contiguous gradients outside the for-loop to guarantee
  // that the contiguous tensors will outlive the kernel computation
  std::vector<c10::MaybeOwned<at::Tensor>> grad_output_contigs;
  grad_output_contigs.reserve(group_size);

  const int cols_per_warp = get_group_index_select_cols_per_warp();
  int64_t idx_small = 0;
  int64_t idx_large = 0;

  for (const auto i : c10::irange(group_size)) {
    const auto& grad = grad_output_group[i];
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(grad, first_indices);

    // Store grad contigs to keep them alive during the kernel computation
    grad_output_contigs.push_back(grad.expect_contiguous());

    // Compute the total number of elements for all grad_inputs
    int64_t grad_input_numel = output_shape_group[i * output_dim];
    for (auto j = (i * output_dim) + 1; j < (i + 1) * output_dim; j++) {
      grad_input_numel *= output_shape_group[j];
    }
    grad_input_numels.push_back(grad_input_numel);
    group_grad_input_numel += grad_input_numel;

    // Put all grad output/input pointers in an array
    if(grad.size(1) < cols_per_warp) {
      grad_output_ptrs_small[idx_small] =
          reinterpret_cast<int64_t>(grad_output_contigs[i]->data_ptr());
          idx_small++;
    } else {
      grad_output_ptrs_large[idx_large] =
          reinterpret_cast<int64_t>(grad_output_contigs[i]->data_ptr());
          idx_large++;
    }
  }

  // Allocate a big tensor to avoid calling many small elementwise kernels
  const auto group_grad_input =
      at::zeros({group_grad_input_numel}, fwd_input.options());
  TORCH_CHECK(group_grad_input.is_contiguous());

  // Split to output_group
  auto output_group = group_grad_input.split(grad_input_numels, 0);

  TORCH_CHECK(output_group.size() == static_cast<size_t>(group_size));

  // Reset the counters of the small and large arguments
  idx_small = 0;
  idx_large = 0;

  // Reshape grad inputs and obtain their pointers
  for (int i = 0; i < group_size; i++) {
    const auto grad_input_shape = std::vector<int64_t>(
        output_shape_group.begin() + i * output_dim,
        output_shape_group.begin() + (i + 1) * output_dim);
    output_group[i] = output_group[i].reshape(grad_input_shape);
    TORCH_CHECK(output_group[i].is_contiguous());

    if(grad_output_group[i].size(1) < cols_per_warp) {
      grad_input_ptrs_small[idx_small] = reinterpret_cast<int64_t>(output_group[i].data_ptr());
      idx_small++;
    } else {
      grad_input_ptrs_large[idx_large] = reinterpret_cast<int64_t>(output_group[i].data_ptr());
      idx_large++;
    }

    // 2) Add group_size gradients for inputs
    outputs.push_back(output_group[i]);
  }

  // Reset the counters of the small and large arguments
  idx_small = 0;
  idx_large = 0;

  // Calculate indices_ptrs
  std::vector<c10::MaybeOwned<at::Tensor>> index_contigs;
  index_contigs.reserve(group_size);
  for (const auto i : c10::irange(group_size)) {
    const auto& indices = indices_group[i];
    index_contigs.push_back(indices.expect_contiguous());
    
    if(grad_output_group[i].size(1) < cols_per_warp) {
      indices_ptrs_small[idx_small] = reinterpret_cast<int64_t>(index_contigs[i]->data_ptr());
      idx_small++;
    } else {
      indices_ptrs_large[idx_large] = reinterpret_cast<int64_t>(index_contigs[i]->data_ptr());
      idx_large++;
    }
  }

  // Transfer grad output pointers to GPU
  args_tensor_small = args_tensor_small.to(first_indices.device(), /*non_blocking=*/true);
  args_tensor_large = args_tensor_large.to(first_indices.device(), /*non_blocking=*/true);

  if(count_small > 0) {
    group_index_select_or_add_cuda_smallEmbD(
      args_tensor_small.data_ptr<int64_t>(),
      args_tensor_small.data_ptr<int64_t>() + count_small,
      args_tensor_small.data_ptr<int64_t>() + 2 * count_small,
      warp_offsets_group_small,
      num_cols_group_small,
      fwd_input.scalar_type(),
      first_indices.scalar_type(),
      fwd_input.device().index(),
      num_input_rows,
      total_num_warps_small,
      group_size,
      /*use_index_select=*/false,
      use_var_cols);
  }

  if(count_large > 0) {
    group_index_select_or_add_cuda(
      args_tensor_large.data_ptr<int64_t>(),
      args_tensor_large.data_ptr<int64_t>() + count_large,
      args_tensor_large.data_ptr<int64_t>() + 2 * count_large,
      warp_offsets_group_large,
      num_cols_group_large,
      fwd_input.scalar_type(),
      first_indices.scalar_type(),
      fwd_input.device().index(),
      num_input_rows,
      total_num_warps_large,
      group_size,
      /*use_index_select=*/false,
      use_var_cols_large);
  }

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
