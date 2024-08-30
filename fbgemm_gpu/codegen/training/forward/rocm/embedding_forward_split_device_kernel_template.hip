#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/hip_kernel_inc/split_embeddings_common.h"

{%- set ddesc = "dense" if dense else "split" %}
{%- set wdesc = "weighted" if weighted else "unweighted" %}
{%- set ndesc = "_nobag" if nobag else "" %}
{%- set vdesc = "_vbe" if vbe else "" %}

template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    typename index_t,
    int32_t embedding_dim,
    int32_t bag_prefetch,
    int32_t bag_unroll,
    bool    weighted>
__device__ void split_tbe_fwd_hip_kernel(
    output_t * p_output,
    const emb_t * p_emb_table,
    const index_t * p_indices,
    const index_t * p_offsets,
    const int32_t * D_offsets,
    const int64_t * weights_offsets,
    const int64_t pooling_mode,
    uint32_t batch,
    uint32_t num_rows,
    uint32_t num_tables,
    const cache_t * p_indice_weights = nullptr)
{ 
    const auto emb_dim = D_offsets[blockIdx.y + 1] - D_offsets[blockIdx.y];
    constexpr uint32_t dword_output_per_row = (embedding_dim + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    // constexpr uint32_t input_data_per_dword = 4 / sizeof(emb_t);    // TODO: larger than 4 byte
    // constexpr uint32_t dword_input_per_row = (dword_output_per_row + input_data_per_dword - 1) / input_data_per_dword;
    // constexpr uint32_t dword_input_per_row_rounded = dword_input_per_row == 1 ? dword_input_per_row
    //                                  : ((dword_input_per_row + 1) / 2 * 2); // round to 2x
    constexpr uint32_t length_mask = ~(bag_unroll - 1);

    static_assert(bag_prefetch < bag_unroll, "");

    // float accumulator[dword_output_per_row];
    output_t accumulator[dword_output_per_row];
    index_t indices[bag_unroll];
    cache_t indice_weights[bag_unroll];

    emb_t emb_data[dword_output_per_row * bag_prefetch];

    int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / AMDGCN_WAVE_SIZE);
    int bag_id = (blockIdx.x << 2) | wave_id;
    // int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    // int bag_id = b_t % batch;
    // printf("bag_id:%d\n", bag_id);
    
    if(bag_id >= batch)
        return ;
    int lane_id = threadIdx.x & (AMDGCN_WAVE_SIZE - 1);

    p_offsets += blockIdx.y * batch + bag_id;
    index_t indices_start = p_offsets[0];
    index_t indices_end = p_offsets[1];

    p_emb_table += weights_offsets[blockIdx.y];
    p_output += D_offsets[blockIdx.y] + bag_id * D_offsets[num_tables];

    #pragma unroll
    for(int i=0; i < dword_output_per_row; i++)
    {
        accumulator[i] = .0f;
    }
    p_indices += indices_start;

    if constexpr (weighted) {
        p_indice_weights += indices_start;
    }

    int32_t length = indices_end - indices_start;
    int32_t length_mod = length & length_mask;

    int itr = 0;
    if(length_mod == 0)
        goto L_end;

    if constexpr (!weighted) {
        #pragma unroll
        for(int i=0; i < bag_unroll; i++){
            indices[i] = p_indices[i];
        }
    } else {
        #pragma unroll
        for(int i=0; i < bag_unroll; i++){
            indices[i] = p_indices[i];
	        indice_weights[i] = p_indice_weights[i];
        }
    }

    itr += bag_unroll;
    p_indices += bag_unroll;

    if constexpr (weighted) {
        p_indice_weights += bag_unroll;
    }

    // LOOP
    for( ; itr<length_mod; itr += bag_unroll){
        load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[0], p_emb_table, lane_id);
        load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[1], p_emb_table, lane_id);

	if constexpr (!weighted) {
            #pragma unroll
            for(int j = 2 ; j < bag_unroll; j += 2){
                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id);
                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[j], p_emb_table, lane_id);

                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id);
                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[j+1], p_emb_table, lane_id);
            }
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0], &emb_data[0], lane_id);
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id);

            #pragma unroll
            for(int i=0; i < bag_unroll; i++){
                indices[i] = p_indices[i];
            }
            p_indices += bag_unroll;

        } else {    // row weighted
            #pragma unroll
            for(int j = 2 ; j < bag_unroll; j += 2){
                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id, indice_weights[j-2]);
                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[j], p_emb_table, lane_id);

                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id, indice_weights[j-1]);
                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[j+1], p_emb_table, lane_id);
            }
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0], &emb_data[0], lane_id, indice_weights[bag_unroll-2]);
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id, indice_weights[bag_unroll-1]);

            #pragma unroll
            for(int i=0; i < bag_unroll; i++){
                indices[i] = p_indices[i];
                indice_weights[i] = p_indice_weights[i];
            }
            p_indices += bag_unroll;
            p_indice_weights += bag_unroll;
        }
    }
    // LAST
    load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[0], p_emb_table, lane_id);
    load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[1], p_emb_table, lane_id);

    if constexpr (!weighted) {
        #pragma unroll
        for(int j = 2 ; j < bag_unroll; j += 2){
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id);
            load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[j], p_emb_table, lane_id);

            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id);
            load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[j+1], p_emb_table, lane_id);
        }
        accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id);
        accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id);

    } else {    // row weighted
        #pragma unroll
        for(int j = 2 ; j < bag_unroll; j += 2){
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id, indice_weights[j-2]);
            load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[j], p_emb_table, lane_id);

            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id, indice_weights[j-1]);
            load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[j+1], p_emb_table, lane_id);
        }
        accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id, indice_weights[bag_unroll-2]);
        accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id, indice_weights[bag_unroll-1]);
    }

L_end:
    if(length & (bag_unroll - 1)){
        if constexpr (!weighted) {
            // last, load one by one
            do {
                indices[0] = p_indices[0];
                p_indices++;

                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[0], p_emb_table, lane_id);
                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id);

                itr++;
            } while (itr < length);
        } else {    // row weighted
            do {
                indices[0] = p_indices[0];
                indice_weights[0] = p_indice_weights[0];
                p_indices++;
                p_indice_weights++;

                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[0], p_emb_table, lane_id);
                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id, indice_weights[0]);

                itr++;
            } while(itr < length);
        }
    }

    if (static_cast<fbgemm_gpu::PoolingMode>(pooling_mode) == fbgemm_gpu::PoolingMode::MEAN && length != 0){
#pragma unroll
        for (int i = 0; i < dword_output_per_row; i++){
            accumulator[i] *= 1.0f / length;
        }
    }

    // store out
    // store_row_per_warp<emb_t, embedding_dim, output_t>::run(&accumulator[0], p_output, lane_id);
    store_row_per_warp<output_t, embedding_dim, output_t>::run(&accumulator[0], p_output, lane_id);
}