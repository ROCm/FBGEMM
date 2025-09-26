/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

#define GROUP_REDUCE_ALL_SUM(val, ...) \
  warpReduceAllSum<__VA_ARGS__, kThreadGroupSize>(val, shfl_sync_mask)

{%- set mdesc = "ssd" if ssd else "split" %}
{%- set locs_or_addrs_tensor = "ssd_row_addrs" if ssd else "lxu_cache_locations" %}
{%- set locs_or_addrs_type = "int64_t" if ssd else "int32_t" %}
{%- set locs_or_addrs_idx = "row_idx" if ssd else "cache_idx" %}

using namespace fbgemm_gpu;

{%- if not is_index_select and optimizer == "rowwise_adagrad" and not dense and not nobag and not weighted and not vbe and not is_gwd_kernel and not ssd %}

template <
    typename emb_t,
    typename cache_t,
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize = kWarpSize,
    int32_t VEC_WIDTH,
    bool kUseVecBlocking
>
DEVICE_INLINE void split_rowwise_adagrad_table_update_kernel(
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& dev_weights,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>& lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& weights_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>& weights_offsets,
    // const int64_t weights_offset,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& sorted_lxu_cache_locations,
    Vec4TAcc<cache_t>* grad_sum,
    Vec4TAcc<cache_t>* smem_grad_sum,
    Vec4TAcc<cache_t>* shared_weight_update_row,
    const bool stochastic_rounding,
    const at::PhiloxCudaState& stochastic_rounding_philox_args,
    const uint32_t run_id,
    const uint32_t cache_loc_run_id,
    const int32_t D,
    const int32_t t,
    const int64_t idx,
    const float global_weight_decay,
    const uint32_t shfl_sync_mask,
    const int32_t max_vecs_per_thread,
    pta::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits>& momentum1_dev,
    pta::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits>& momentum1_uvm,
    pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& momentum1_placements,
    pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>& momentum1_offsets,
    // const int64_t momentum1_offset,
    float learning_rate = 0,
    float eps = 0,
    float weight_decay = 0.0,
    int64_t weight_decay_mode = 0,
    float max_norm = 0.0
) {
    constexpr auto kIsInt8 = std::is_same_v<emb_t, uint8_t>;
    // Copy value to max_vecs to make max_vecs_per_thread known at compile time
    // when kUseVecBlocking == false
    const int32_t max_vecs =
        kUseVecBlocking ? max_vecs_per_thread : kFixedMaxVecsPerThread;
    const int64_t weights_offset = weights_offsets[t];
    emb_t* __restrict__ weights {nullptr};
    cache_t* __restrict__ cache_weights {nullptr};
    int32_t D_emb = D;
    if constexpr (kIsInt8) {
        D_emb += kINT8QparamsBytes;
    }
    const auto weights_placement = static_cast<PlacementType>(weights_placements[t]);
    if (weights_placement == PlacementType::DEVICE) {
        weights = &dev_weights[weights_offset + idx * D_emb];
    } else {
        weights = &uvm_weights[weights_offset + idx * D_emb];
    }
    if (weights_placement == PlacementType::MANAGED_CACHING) {
        const auto cache_idx = sorted_lxu_cache_locations[cache_loc_run_id];
        if (cache_idx != kCacheLocationMissing) {
          cache_weights = &lxu_cache_weights[cache_idx][0];
        }
    }
    at::acc_type<cache_t, true>* __restrict__ momentum1;
    const auto momentum1_placement = static_cast<PlacementType>(momentum1_placements[t]);
    const int64_t momentum1_offset = momentum1_offsets[t];
    if (momentum1_placement == PlacementType::DEVICE) {
        momentum1 = &momentum1_dev[momentum1_offset];
    } else {
        momentum1 = &momentum1_uvm[momentum1_offset];
    }

    auto weight_row_template =
        WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
            weights,
            cache_weights,
            D,
            stochastic_rounding,
            &stochastic_rounding_philox_args,
            threadIdx.x + run_id * blockDim.x);

    float2 qparams_template;
    if constexpr (kIsInt8) {
        if (!cache_weights) {
            qparams_template = weight_row_template.load_qparams();
        }
    }
    [[maybe_unused]] constexpr auto enable_optimizer_offloading = false;


    at::acc_type<cache_t, true> g_local_sum_square = 0.0;

    if constexpr (kUseVecBlocking) {
        // max_vecs is not known at compile time
        for (int32_t vec = 0;
            vec < max_vecs &&
            (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
            ++vec) {
            const int32_t d_vec = vec * kThreadGroupSize + threadIdx.x;
            [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;

        const float4* grad = &smem_grad_sum[d_vec].acc;
        auto gx = grad->x;
        auto gy = grad->y;
        auto gz = grad->z;
        auto gw = grad->w;
        if (weight_decay_mode == 1) {
            // L2 regularization
            Vec4TAcc<cache_t> weight = weight_row_template.load(d, qparams_template);
            gx += weight_decay * weight.acc.x;
            gy += weight_decay * weight.acc.y;
            gz += weight_decay * weight.acc.z;
            gw += weight_decay * weight.acc.w;
        }
        g_local_sum_square += gx * gx + gy * gy + gz * gz + gw * gw;

        }

    } else {
        // kFixedMaxVecsPerThread is known at compile time
        #pragma unroll kFixedMaxVecsPerThread
        for (int32_t vec = 0;
            vec < kFixedMaxVecsPerThread
                && (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
            ++vec) {
            const int32_t d_vec = vec * kThreadGroupSize + threadIdx.x;
            [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;

        const float4* grad = &grad_sum[vec].acc;
        auto gx = grad->x;
        auto gy = grad->y;
        auto gz = grad->z;
        auto gw = grad->w;
        if (weight_decay_mode == 1) {
            // L2 regularization
            Vec4TAcc<cache_t> weight = weight_row_template.load(d, qparams_template);
            gx += weight_decay * weight.acc.x;
            gy += weight_decay * weight.acc.y;
            gz += weight_decay * weight.acc.z;
            gw += weight_decay * weight.acc.w;
        }
        g_local_sum_square += gx * gx + gy * gy + gz * gz + gw * gw;

        }
    }

	// Define the rowwise adagrad optimizer state struct view
    struct [[maybe_unused]] OptimizerState {
        at::acc_type<cache_t, true> momentum;
    };

    const at::acc_type<cache_t, true> g_avg_square =
        GROUP_REDUCE_ALL_SUM(g_local_sum_square, at::acc_type<cache_t, true>) / D;

    at::acc_type<cache_t, true> multiplier = 0.0;
    at::acc_type<cache_t, true> correction = 0.0;
    if (threadIdx.x == 0) {	
        auto new_sum_square_grads = g_avg_square;

        // Update the optimizer state.  Use optimizer state offloading only if 
        // SSD and if enabled by the user
        if (enable_optimizer_offloading) {
            // Fetch the pointer to the optimizer state along the cache row
            auto* optimizer = weight_row_template.template optimizer_state_ptr<OptimizerState>();
            new_sum_square_grads += optimizer->momentum;
            optimizer->momentum = new_sum_square_grads;

        } else {
            new_sum_square_grads += momentum1[idx];
            momentum1[idx] = new_sum_square_grads;
        }

        multiplier = learning_rate / (sqrtf(new_sum_square_grads) + eps);
        if (weight_decay_mode == 1) {
            // L2 regularization
            correction = 1.0 - multiplier * weight_decay;
        } else if (weight_decay_mode == 2 || weight_decay_mode == 5) {
            // Decoupled weight decay
            correction = 1.0 - learning_rate * weight_decay;
        } else {
            // default value
            correction = 1.0;
        }
    }
    multiplier = SHFL_SYNC(multiplier, 0);
    correction = SHFL_SYNC(correction, 0);



    float2 qparams_new;

    if constexpr (kUseVecBlocking) {
        // max_vecs is not known at compile time
        for (int32_t vec = 0;
            vec < max_vecs &&
            (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
            ++vec) {
            const int32_t d_vec = vec * kThreadGroupSize + threadIdx.x;
            [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;

           Vec4TAcc<cache_t> weight_new = weight_row_template.load(d, qparams_template);
           Vec4TAcc<cache_t>& grad = smem_grad_sum[d_vec];
           weight_new.mul_(global_weight_decay);

        weight_new.acc.x = correction * weight_new.acc.x - multiplier * grad.acc.x;
        weight_new.acc.y = correction * weight_new.acc.y - multiplier * grad.acc.y;
        weight_new.acc.z = correction * weight_new.acc.z - multiplier * grad.acc.z;
        weight_new.acc.w = correction * weight_new.acc.w - multiplier * grad.acc.w;

           if (kIsInt8 && !cache_weights) {
               shared_weight_update_row[d_vec] = weight_new;
           } else {
               // qparams_new not used if type is not int8
               weight_row_template.store(weight_new, d, qparams_new);
           }

        }

    } else {
        // kFixedMaxVecsPerThread is known at compile time
        #pragma unroll kFixedMaxVecsPerThread
        for (int32_t vec = 0;
            vec < kFixedMaxVecsPerThread
                && (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
            ++vec) {
            const int32_t d_vec = vec * kThreadGroupSize + threadIdx.x;
            [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;

           Vec4TAcc<cache_t> weight_new = weight_row_template.load(d, qparams_template);
           Vec4TAcc<cache_t>& grad = grad_sum[vec];
           weight_new.mul_(global_weight_decay);

        weight_new.acc.x = correction * weight_new.acc.x - multiplier * grad.acc.x;
        weight_new.acc.y = correction * weight_new.acc.y - multiplier * grad.acc.y;
        weight_new.acc.z = correction * weight_new.acc.z - multiplier * grad.acc.z;
        weight_new.acc.w = correction * weight_new.acc.w - multiplier * grad.acc.w;

           if (kIsInt8 && !cache_weights) {
               shared_weight_update_row[d_vec] = weight_new;
           } else {
               // qparams_new not used if type is not int8
               weight_row_template.store(weight_new, d, qparams_new);
           }

        }
    }


    if constexpr (kIsInt8) {
        if (!cache_weights) {
            // Calculate new qparams after row update
            qparams_new = thrust_find_qparams<at::acc_type<cache_t, true>>(
                shared_weight_update_row, D);
            weight_row_template.store_qparams(qparams_new);

            // Fetch cached updated row from shared mem and quantize on-the-fly
            // when saving to lowp embedding
            for (int32_t vec = 0;
                (vec * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++vec) {
                const auto d_vec = vec * kThreadGroupSize + threadIdx.x;
                const int32_t d = d_vec * VEC_WIDTH;
                weight_row_template.store(
                    shared_weight_update_row[d_vec],
                    d,
                    qparams_new);
            }
        }
    }


    if (max_norm > 0.0) {
        CUDA_KERNEL_ASSERT(!(std::is_same<emb_t, uint8_t>::value && !cache_weights)); // not supported for uint8 yet

        // compute weight norm
        at::acc_type<cache_t, true> weight_sum_square = 0.0;
        for (int32_t vec = 0;
             vec < max_vecs && (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
             ++vec) {
            const int32_t d = (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH;
            Vec4TAcc<cache_t> weight_new = weight_row_template.load(d, qparams_template);
            weight_sum_square
                += weight_new.acc.x * weight_new.acc.x
                + weight_new.acc.y * weight_new.acc.y
                + weight_new.acc.z * weight_new.acc.z
                + weight_new.acc.w * weight_new.acc.w;
        }
        const at::acc_type<cache_t, true> weight_norm =
            sqrtf(GROUP_REDUCE_ALL_SUM(weight_sum_square, at::acc_type<cache_t, true>));

        // scale by max_norm if weight_norm exceeds max_norm
        if (threadIdx.x == 0) {
            multiplier = weight_norm > max_norm ? max_norm / weight_norm : 1.0f;
        }
        multiplier = SHFL_SYNC(multiplier, 0);
        if (weight_norm > max_norm) {
            for (int32_t vec = 0;
                 vec < max_vecs && (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
                 ++vec) {
                const int32_t d = (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH;
                Vec4TAcc<cache_t> weight_new = weight_row_template.load(d, qparams_template);

                weight_new.acc.x *= multiplier;
                weight_new.acc.y *= multiplier;
                weight_new.acc.z *= multiplier;
                weight_new.acc.w *= multiplier;
                weight_row_template.store(weight_new, d, qparams_new); // qparams_new not used if embedding is not int8
            }
        }
    }

}

template <
    typename emb_t,
    typename cache_t,
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize = kWarpSize,
    int32_t VEC_WIDTH,
    bool kUseVecBlocking
>
DEVICE_INLINE void split_rowwise_adagrad_table_update_kernel_device(
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& dev_weights,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>& lxu_cache_weights,
    // const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& weights_placements,
    // const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>& weights_offsets,
    PlacementType weights_placement,
    const int64_t weights_offset,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& sorted_lxu_cache_locations,
    Vec4TAcc<cache_t>* grad_sum,
    Vec4TAcc<cache_t>* smem_grad_sum,
    Vec4TAcc<cache_t>* shared_weight_update_row,
    const bool stochastic_rounding,
    const at::PhiloxCudaState& stochastic_rounding_philox_args,
    const uint32_t run_id,
    const uint32_t cache_loc_run_id,
    const int32_t D,
    const int32_t t,
    const int64_t idx,
    const float global_weight_decay,
    const uint32_t shfl_sync_mask,
    const int32_t max_vecs_per_thread,
    pta::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits>& momentum1_dev,
    pta::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits>& momentum1_uvm,
    // pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& momentum1_placements,
    // pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>& momentum1_offsets,
    PlacementType momentum1_placement,
    const int64_t momentum1_offset,
    float learning_rate = 0,
    float eps = 0,
    float weight_decay = 0.0,
    int64_t weight_decay_mode = 0,
    float max_norm = 0.0
) {
    constexpr auto kIsInt8 = std::is_same_v<emb_t, uint8_t>;
    // Copy value to max_vecs to make max_vecs_per_thread known at compile time
    // when kUseVecBlocking == false
    const int32_t max_vecs =
        kUseVecBlocking ? max_vecs_per_thread : kFixedMaxVecsPerThread;
    // const int64_t weights_offset = weights_offsets[t];
    emb_t* __restrict__ weights {nullptr};
    cache_t* __restrict__ cache_weights {nullptr};
    int32_t D_emb = D;
    if constexpr (kIsInt8) {
        D_emb += kINT8QparamsBytes;
    }
    // const auto weights_placement = static_cast<PlacementType>(weights_placements[t]);
    if (weights_placement == PlacementType::DEVICE) {
        weights = &dev_weights[weights_offset + idx * D_emb];
    } else {
        weights = &uvm_weights[weights_offset + idx * D_emb];
    }
    if (weights_placement == PlacementType::MANAGED_CACHING) {
        const auto cache_idx = sorted_lxu_cache_locations[cache_loc_run_id];
        if (cache_idx != kCacheLocationMissing) {
          cache_weights = &lxu_cache_weights[cache_idx][0];
        }
    }
    at::acc_type<cache_t, true>* __restrict__ momentum1;
    // const auto momentum1_placement = static_cast<PlacementType>(momentum1_placements[t]);
    // const int64_t momentum1_offset = momentum1_offsets[t];
    if (momentum1_placement == PlacementType::DEVICE) {
        momentum1 = &momentum1_dev[momentum1_offset];
    } else {
        momentum1 = &momentum1_uvm[momentum1_offset];
    }

    auto weight_row_template =
        WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
            weights,
            cache_weights,
            D,
            stochastic_rounding,
            &stochastic_rounding_philox_args,
            threadIdx.x + run_id * blockDim.x);

    float2 qparams_template;
    if constexpr (kIsInt8) {
        if (!cache_weights) {
            qparams_template = weight_row_template.load_qparams();
        }
    }
    [[maybe_unused]] constexpr auto enable_optimizer_offloading = false;


    at::acc_type<cache_t, true> g_local_sum_square = 0.0;

    if constexpr (kUseVecBlocking) {
        // max_vecs is not known at compile time
        for (int32_t vec = 0;
            vec < max_vecs &&
            (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
            ++vec) {
            const int32_t d_vec = vec * kThreadGroupSize + threadIdx.x;
            [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;

        const float4* grad = &smem_grad_sum[d_vec].acc;
        auto gx = grad->x;
        auto gy = grad->y;
        auto gz = grad->z;
        auto gw = grad->w;
        if (weight_decay_mode == 1) {
            // L2 regularization
            Vec4TAcc<cache_t> weight = weight_row_template.load(d, qparams_template);
            gx += weight_decay * weight.acc.x;
            gy += weight_decay * weight.acc.y;
            gz += weight_decay * weight.acc.z;
            gw += weight_decay * weight.acc.w;
        }
        g_local_sum_square += gx * gx + gy * gy + gz * gz + gw * gw;

        }

    } else {
        // kFixedMaxVecsPerThread is known at compile time
        #pragma unroll kFixedMaxVecsPerThread
        for (int32_t vec = 0;
            vec < kFixedMaxVecsPerThread
                && (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
            ++vec) {
            const int32_t d_vec = vec * kThreadGroupSize + threadIdx.x;
            [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;

        const float4* grad = &grad_sum[vec].acc;
        auto gx = grad->x;
        auto gy = grad->y;
        auto gz = grad->z;
        auto gw = grad->w;
        if (weight_decay_mode == 1) {
            // L2 regularization
            Vec4TAcc<cache_t> weight = weight_row_template.load(d, qparams_template);
            gx += weight_decay * weight.acc.x;
            gy += weight_decay * weight.acc.y;
            gz += weight_decay * weight.acc.z;
            gw += weight_decay * weight.acc.w;
        }
        g_local_sum_square += gx * gx + gy * gy + gz * gz + gw * gw;

        }
    }

	// Define the rowwise adagrad optimizer state struct view
    struct [[maybe_unused]] OptimizerState {
        at::acc_type<cache_t, true> momentum;
    };

    const at::acc_type<cache_t, true> g_avg_square =
        GROUP_REDUCE_ALL_SUM(g_local_sum_square, at::acc_type<cache_t, true>) / D;

    at::acc_type<cache_t, true> multiplier = 0.0;
    at::acc_type<cache_t, true> correction = 0.0;
    if (threadIdx.x == 0) {	
        auto new_sum_square_grads = g_avg_square;

        // Update the optimizer state.  Use optimizer state offloading only if 
        // SSD and if enabled by the user
        if (enable_optimizer_offloading) {
            // Fetch the pointer to the optimizer state along the cache row
            auto* optimizer = weight_row_template.template optimizer_state_ptr<OptimizerState>();
            new_sum_square_grads += optimizer->momentum;
            optimizer->momentum = new_sum_square_grads;

        } else {
            new_sum_square_grads += momentum1[idx];
            momentum1[idx] = new_sum_square_grads;
        }

        multiplier = learning_rate / (sqrtf(new_sum_square_grads) + eps);
        if (weight_decay_mode == 1) {
            // L2 regularization
            correction = 1.0 - multiplier * weight_decay;
        } else if (weight_decay_mode == 2 || weight_decay_mode == 5) {
            // Decoupled weight decay
            correction = 1.0 - learning_rate * weight_decay;
        } else {
            // default value
            correction = 1.0;
        }
    }
    multiplier = SHFL_SYNC(multiplier, 0);
    correction = SHFL_SYNC(correction, 0);



    float2 qparams_new;

    if constexpr (kUseVecBlocking) {
        // max_vecs is not known at compile time
        for (int32_t vec = 0;
            vec < max_vecs &&
            (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
            ++vec) {
            const int32_t d_vec = vec * kThreadGroupSize + threadIdx.x;
            [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;

           Vec4TAcc<cache_t> weight_new = weight_row_template.load(d, qparams_template);
           Vec4TAcc<cache_t>& grad = smem_grad_sum[d_vec];
           weight_new.mul_(global_weight_decay);

        weight_new.acc.x = correction * weight_new.acc.x - multiplier * grad.acc.x;
        weight_new.acc.y = correction * weight_new.acc.y - multiplier * grad.acc.y;
        weight_new.acc.z = correction * weight_new.acc.z - multiplier * grad.acc.z;
        weight_new.acc.w = correction * weight_new.acc.w - multiplier * grad.acc.w;

           if (kIsInt8 && !cache_weights) {
               shared_weight_update_row[d_vec] = weight_new;
           } else {
               // qparams_new not used if type is not int8
               weight_row_template.store(weight_new, d, qparams_new);
           }

        }

    } else {
        // kFixedMaxVecsPerThread is known at compile time
        #pragma unroll kFixedMaxVecsPerThread
        for (int32_t vec = 0;
            vec < kFixedMaxVecsPerThread
                && (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
            ++vec) {
            const int32_t d_vec = vec * kThreadGroupSize + threadIdx.x;
            [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;

           Vec4TAcc<cache_t> weight_new = weight_row_template.load(d, qparams_template);
           Vec4TAcc<cache_t>& grad = grad_sum[vec];
           weight_new.mul_(global_weight_decay);

        weight_new.acc.x = correction * weight_new.acc.x - multiplier * grad.acc.x;
        weight_new.acc.y = correction * weight_new.acc.y - multiplier * grad.acc.y;
        weight_new.acc.z = correction * weight_new.acc.z - multiplier * grad.acc.z;
        weight_new.acc.w = correction * weight_new.acc.w - multiplier * grad.acc.w;

           if (kIsInt8 && !cache_weights) {
               shared_weight_update_row[d_vec] = weight_new;
           } else {
               // qparams_new not used if type is not int8
               weight_row_template.store(weight_new, d, qparams_new);
           }

        }
    }


    if constexpr (kIsInt8) {
        if (!cache_weights) {
            // Calculate new qparams after row update
            qparams_new = thrust_find_qparams<at::acc_type<cache_t, true>>(
                shared_weight_update_row, D);
            weight_row_template.store_qparams(qparams_new);

            // Fetch cached updated row from shared mem and quantize on-the-fly
            // when saving to lowp embedding
            for (int32_t vec = 0;
                (vec * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++vec) {
                const auto d_vec = vec * kThreadGroupSize + threadIdx.x;
                const int32_t d = d_vec * VEC_WIDTH;
                weight_row_template.store(
                    shared_weight_update_row[d_vec],
                    d,
                    qparams_new);
            }
        }
    }


    if (max_norm > 0.0) {
        CUDA_KERNEL_ASSERT(!(std::is_same<emb_t, uint8_t>::value && !cache_weights)); // not supported for uint8 yet

        // compute weight norm
        at::acc_type<cache_t, true> weight_sum_square = 0.0;
        for (int32_t vec = 0;
             vec < max_vecs && (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
             ++vec) {
            const int32_t d = (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH;
            Vec4TAcc<cache_t> weight_new = weight_row_template.load(d, qparams_template);
            weight_sum_square
                += weight_new.acc.x * weight_new.acc.x
                + weight_new.acc.y * weight_new.acc.y
                + weight_new.acc.z * weight_new.acc.z
                + weight_new.acc.w * weight_new.acc.w;
        }
        const at::acc_type<cache_t, true> weight_norm =
            sqrtf(GROUP_REDUCE_ALL_SUM(weight_sum_square, at::acc_type<cache_t, true>));

        // scale by max_norm if weight_norm exceeds max_norm
        if (threadIdx.x == 0) {
            multiplier = weight_norm > max_norm ? max_norm / weight_norm : 1.0f;
        }
        multiplier = SHFL_SYNC(multiplier, 0);
        if (weight_norm > max_norm) {
            for (int32_t vec = 0;
                 vec < max_vecs && (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
                 ++vec) {
                const int32_t d = (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH;
                Vec4TAcc<cache_t> weight_new = weight_row_template.load(d, qparams_template);

                weight_new.acc.x *= multiplier;
                weight_new.acc.y *= multiplier;
                weight_new.acc.z *= multiplier;
                weight_new.acc.w *= multiplier;
                weight_row_template.store(weight_new, d, qparams_new); // qparams_new not used if embedding is not int8
            }
        }
    }

}

{%- else %}

template <
    typename emb_t,
    typename cache_t,
    {%- for ph_name in args.placeholder_tensor_names %}
    {%- set ph_type = "{}_ph_t".format(ph_name) %}
    typename {{ ph_type }},
    {%- endfor %}
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize = kWarpSize,
    int32_t VEC_WIDTH,
    bool kUseVecBlocking
>
DEVICE_INLINE void {{ mdesc }}_{{ optimizer }}_table_update_kernel(
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& dev_weights,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>& lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& weights_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>& weights_offsets,
    const pta::PackedTensorAccessor32<{{ locs_or_addrs_type }}, 1, at::RestrictPtrTraits>& sorted_{{ locs_or_addrs_tensor }},
    Vec4TAcc<cache_t>* grad_sum,
    Vec4TAcc<cache_t>* smem_grad_sum,
    Vec4TAcc<cache_t>* shared_weight_update_row,
    const bool stochastic_rounding,
    const at::PhiloxCudaState& stochastic_rounding_philox_args,
    const uint32_t run_id,
    const uint32_t cache_loc_run_id,
    const int32_t D,
    const int32_t t,
    const int64_t idx,
    {%- if has_global_weight_decay_support %}
    const float global_weight_decay,
    {%- endif %}
    const uint32_t shfl_sync_mask,
    const int32_t max_vecs_per_thread,
    {%- if ssd %}
    const bool enable_optimizer_offloading,
    {%- endif %}
    {{ args.split_ref_kernel_args | replace_pta_namespace() | join(",\n    ") }}
) {
    constexpr auto kIsInt8 = std::is_same_v<emb_t, uint8_t>;
    // Copy value to max_vecs to make max_vecs_per_thread known at compile time
    // when kUseVecBlocking == false
    const int32_t max_vecs =
        kUseVecBlocking ? max_vecs_per_thread : kFixedMaxVecsPerThread;
    const int64_t weights_offset = weights_offsets[t];
    emb_t* __restrict__ weights {nullptr};
    cache_t* __restrict__ cache_weights {nullptr};
    int32_t D_emb = D;
    if constexpr (kIsInt8) {
        D_emb += kINT8QparamsBytes;
    }
    const auto weights_placement = static_cast<PlacementType>(weights_placements[t]);
    if (weights_placement == PlacementType::DEVICE) {
        weights = &dev_weights[weights_offset + idx * D_emb];
    } else {
        weights = {{ "nullptr" if ssd else "&uvm_weights[weights_offset + idx * D_emb]" }};
    }
    if (weights_placement == PlacementType::MANAGED_CACHING) {
        const auto {{ locs_or_addrs_idx }} = sorted_{{ locs_or_addrs_tensor }}[cache_loc_run_id];
        {%- if ssd %}
        cache_weights = reinterpret_cast<cache_t*>(
            *reinterpret_cast<const uint64_t*>(&{{ locs_or_addrs_idx }}));
        {%- else %}
        if ({{ locs_or_addrs_idx }} != kCacheLocationMissing) {
          cache_weights = &lxu_cache_weights[{{ locs_or_addrs_idx }}][0];
        }
        {%- endif %}
    }
    {%- for tensor in args.split_tensors %}
    {{ args.split_tensor_types[tensor] }}* __restrict__ {{ tensor }};
    const auto {{ tensor }}_placement = static_cast<PlacementType>({{ tensor }}_placements[t]);
    const int64_t {{ tensor }}_offset = {{ tensor }}_offsets[t];
    if ({{ tensor }}_placement == PlacementType::DEVICE) {
        {{ tensor }} = &{{ tensor }}_dev[{{ tensor }}_offset];
    } else {
        {{ tensor }} = &{{ tensor }}_uvm[{{ tensor }}_offset];
    }
    {%- endfor %}

    auto weight_row_template =
        WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
            weights,
            cache_weights,
            D,
            stochastic_rounding,
            &stochastic_rounding_philox_args,
            threadIdx.x + run_id * blockDim.x);

    float2 qparams_template;
    if constexpr (kIsInt8) {
        if (!cache_weights) {
            qparams_template = weight_row_template.load_qparams();
        }
    }

    {%- if not ssd %}
    [[maybe_unused]] constexpr auto enable_optimizer_offloading = false;
    {%- endif %}

    {{ split_precomputation }}

    {# /* Note: technically, global weight decay (gwd) compensation should be done before
    `split_precomputation`). But since decouple mode in `rowwise_adagrad` only computes correction,
    the order of applying gwd does not matter. We perform gwd update before `split_weight_update`
    below to minimize number of times to load weights.
    So, note that the behavior may be different if you want to enable gwd for other optimizers
    such as `lamb` or `partial_rowwise_lamb`.
    */#}
    float2 qparams_new;
    {{
       generate_optimized_grad_sum_loop_access(
           """
           Vec4TAcc<cache_t> weight_new = weight_row_template.load(d, qparams_template);
           Vec4TAcc<cache_t>& grad = {grad_vec};
           {global_weight_decay_update}
           {split_weight_update}
           if (kIsInt8 && !cache_weights) {
               shared_weight_update_row[d_vec] = weight_new;
           } else {
               // qparams_new not used if type is not int8
               weight_row_template.store(weight_new, d, qparams_new);
           }
           """,
           other_formats={
               "split_weight_update": split_weight_update,
               "global_weight_decay_update": "weight_new.mul_(global_weight_decay);" if has_global_weight_decay_support else ""
            },
       )
    }}

    if constexpr (kIsInt8) {
        if (!cache_weights) {
            // Calculate new qparams after row update
            qparams_new = thrust_find_qparams<at::acc_type<cache_t, true>>(
                shared_weight_update_row, D);
            weight_row_template.store_qparams(qparams_new);

            // Fetch cached updated row from shared mem and quantize on-the-fly
            // when saving to lowp embedding
            for (int32_t vec = 0;
                (vec * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++vec) {
                const auto d_vec = vec * kThreadGroupSize + threadIdx.x;
                const int32_t d = d_vec * VEC_WIDTH;
                weight_row_template.store(
                    shared_weight_update_row[d_vec],
                    d,
                    qparams_new);
            }
        }
    }

    {{ split_post_update }}
}

{%- endif %}
// clang-format on
