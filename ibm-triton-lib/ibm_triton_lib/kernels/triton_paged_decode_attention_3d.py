#  /*******************************************************************************
#   * Copyright 2025 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#

from typing import List, Optional, Tuple, Union, NamedTuple

import os
import torch
import triton
import triton.language as tl

from ..utils.triton_utils import unpack_grid

gpu_name = torch.cuda.get_device_name()
debug_flag = os.getenv("TRITON_BACKEND_DEBUG") == "1"


def metadata_fn(
    grid: tuple,
    metadata: NamedTuple,
    args: dict,
):
    grid_x, grid_y, grid_z = unpack_grid(grid)
    num_warps = metadata.num_warps
    num_stages = metadata.num_stages
    cluster_x, cluster_y, cluster_z = metadata.cluster_dims
    shared_memory = metadata.shared
    # args just contains NON-CONSTANT arguments
    num_seqs, num_query_heads, head_size = args["query_ptr"].shape
    if len(args["key_cache_ptr"].shape) == 5:
        num_blocks, num_kv_heads, _, block_size, cache_align_x = args[
            "key_cache_ptr"
        ].shape
    else:
        num_blocks, num_kv_heads, _, block_size = args["key_cache_ptr"].shape
    _, max_num_blocks_per_seq = args["block_tables_ptr"].shape
    # num tokens are treated as batch
    dtype_size = args["query_ptr"].element_size()

    num_bytes = (
        (dtype_size * num_seqs * num_query_heads * head_size)
        + (dtype_size * num_blocks * num_kv_heads * head_size * block_size * 2)
        + num_seqs * max_num_blocks_per_seq * dtype_size  # dtype size? not ptr size?
    )
    num_flops = num_blocks * num_kv_heads * head_size * block_size * 7  # TODO?
    return {
        "name": f"triton_zrl_paged_attention_3d_<grid:{grid_x}x{grid_y}x{grid_z}>_<cluster:{cluster_x}x{cluster_y}x{cluster_z}>_<warps:{num_warps}>_<shared:{shared_memory}>_<stages:{num_stages}>",
        "flops16": num_flops,
        "bytes": num_bytes,
    }


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit(launch_metadata=metadata_fn)
def kernel_paged_attention_3d(
    segm_output_ptr,  # [num_seqs, num_query_heads, num_segments, head_size]
    segm_max_ptr,  # [num_seqs, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_seqs, num_query_heads, num_segments]
    query_ptr,  # [num_seqs, num_query_heads, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size, block_size]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    scale,  # float32
    cu_q_len_ptr, # [num_seqs+1]
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.constexpr,  # int, should be equal to max_num_blocks_per_seq
    query_stride_0: tl.constexpr,  # int
    query_stride_1: tl.constexpr,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    BLOCKS_PER_SEGMENT: tl.constexpr,  # int
    MAX_NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    x: tl.constexpr,
    stride_k_cache_0: tl.constexpr,
    stride_k_cache_1: tl.constexpr,
    stride_k_cache_2: tl.constexpr,
    stride_k_cache_3: tl.constexpr,
    stride_k_cache_4: tl.constexpr,
    stride_v_cache_0: tl.constexpr,
    stride_v_cache_1: tl.constexpr,
    stride_v_cache_2: tl.constexpr,
    stride_v_cache_3: tl.constexpr,
    filter_by_query_len: tl.constexpr, # bool
    query_start_len_ptr, # [num_seqs+1]
):
    seq_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    # context len for this sequence
    context_len = tl.load(context_lens_ptr + seq_idx)

    if segm_idx * BLOCKS_PER_SEGMENT * BLOCK_SIZE >= context_len:
        return

    # CHECK order of above/below exit conditions

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = (cur_batch_in_all_stop_index -
                               cur_batch_in_all_start_index)

        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    kv_head_idx = query_head_idx // num_queries_per_kv

    query_offset = cur_batch_in_all_start_index * query_stride_0 + query_head_idx * query_stride_1

    # Q : (HEAD_SIZE,)
    Q = tl.load(query_ptr + query_offset + tl.arange(0, HEAD_SIZE))

    block_table_offset = seq_idx * block_table_stride

    m = tl.full([1], float("-inf"), dtype=tl.float32)
    l = tl.full([1], 1.0, dtype=tl.float32)
    acc = tl.zeros([HEAD_SIZE], dtype=tl.float32)

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_head_idx)

    num_blocks = cdiv_fn(context_len, BLOCK_SIZE)

    # iterate through blocks within current segment
    for j in range(
        segm_idx * BLOCKS_PER_SEGMENT,
        min((segm_idx + 1) * BLOCKS_PER_SEGMENT, num_blocks),
    ):
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        offs_n = tl.arange(0, BLOCK_SIZE)
        offs_d = tl.arange(0, HEAD_SIZE)

        v_offset = (physical_block_idx * stride_v_cache_0 +
                    kv_head_idx * stride_v_cache_1 +
                    offs_d[:, None] * stride_v_cache_2 +
                    offs_n[None, :] * stride_v_cache_3)

        k_offset = (physical_block_idx * stride_k_cache_0 +
                    kv_head_idx * stride_k_cache_1 +
                    (offs_d[:, None] // x) * stride_k_cache_2 +
                    offs_n[None, :] * stride_k_cache_3 +
                    (offs_d[:, None] % x) * stride_k_cache_4)

        # K : (HEAD_SIZE, BLOCK_SIZE)
        K = tl.load(key_cache_ptr + k_offset)

        # V : (HEAD_SIZE, BLOCK_SIZE)
        V = tl.load(value_cache_ptr + v_offset)

        tmp = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], context_len, dtype=tl.int32)
        mask_new = tmp < boundary
        # S : (BLOCK_SIZE,)
        S = tl.where(mask_new, 0.0, float("-inf")).to(tl.float32)
        S += scale * tl.sum(K * Q[:, None], axis=0)

        if USE_ALIBI_SLOPES:
            S += alibi_slope * (tmp - context_len + 1)

        # compute running maximum
        # m_j : (1,)
        m_j = tl.maximum(m, tl.max(S, axis=0))

        # P : (BLOCK_SIZE,)
        P = tl.exp(S - m_j)

        # l_j : (1,)
        l_j = tl.sum(P, axis=0)

        # alpha : (1, )
        alpha = tl.exp(m - m_j)

        # acc : (BLOCK_SIZE,)
        acc = acc * alpha

        # update constants
        l = l * alpha + l_j
        m = m_j

        # acc : (BLOCK_SIZE,)
        acc += tl.sum(V * P[None, :], axis=1)

    segm_output_offset = (
        seq_idx * (num_query_heads * MAX_NUM_SEGMENTS_PER_SEQ * HEAD_SIZE)
        + query_head_idx * (MAX_NUM_SEGMENTS_PER_SEQ * HEAD_SIZE)
        + segm_idx * HEAD_SIZE
        + tl.arange(0, HEAD_SIZE)
    )
    tl.store(segm_output_ptr + segm_output_offset, acc)

    segm_offset = (
        seq_idx * (num_query_heads * MAX_NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * MAX_NUM_SEGMENTS_PER_SEQ
        + segm_idx
        + tl.arange(0, 1)
    )
    tl.store(segm_max_ptr + segm_offset, m)
    tl.store(segm_expsum_ptr + segm_offset, l)


@triton.jit
def reduce_segments(
    output_ptr,  # [num_seqs, num_query_heads, head_size]
    segm_output_ptr,  # [num_seqs, num_query_heads, num_segments, head_size]
    segm_max_ptr,  # [num_seqs, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_seqs, num_query_heads, num_segments]
    context_lens_ptr,  # [num_seqs]
    num_query_heads: tl.constexpr,  # int
    output_stride_0: tl.constexpr,  # int
    output_stride_1: tl.constexpr,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    BLOCKS_PER_SEGMENT: tl.constexpr,  # int
    MAX_NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    filter_by_query_len: tl.constexpr, # bool
    query_start_len_ptr, # [num_seqs+1]
):
    seq_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = (cur_batch_in_all_stop_index -
                               cur_batch_in_all_start_index)

        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    # create mask for subsequent loads
    context_len = tl.load(context_lens_ptr + seq_idx)
    act_num_segments = cdiv_fn(context_len, BLOCKS_PER_SEGMENT * BLOCK_SIZE)
    mask = tl.arange(0, MAX_NUM_SEGMENTS_PER_SEQ) < tl.full(
        [MAX_NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )

    # load segment maxima
    segm_offset = (
        seq_idx * (num_query_heads * MAX_NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * MAX_NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, MAX_NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        seq_idx * (num_query_heads * MAX_NUM_SEGMENTS_PER_SEQ * HEAD_SIZE)
        + query_head_idx * (MAX_NUM_SEGMENTS_PER_SEQ * HEAD_SIZE)
        + tl.arange(0, MAX_NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE
        + tl.arange(0, HEAD_SIZE)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset, mask=mask[:, None], other=0.0
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc = tl.sum(segm_output, axis=0) / overall_expsum

    # write result
    output_offset = (
        cur_batch_in_all_start_index * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE)
    )
    tl.store(output_ptr + output_offset, acc)


def paged_attention_triton_3d(
    output,
    query,
    key_cache,
    value_cache,
    scale,
    block_tables,
    context_lens,
    alibi_slopes,
    block_size,
    num_seqs,
    num_query_heads,
    num_queries_per_kv,
    head_size,
    cu_q_len,
):
    blocks_per_segment = 16
    max_num_segments_per_seq = 1
    while max_num_segments_per_seq < (
        (block_tables.stride(0) + blocks_per_segment - 1) // blocks_per_segment
    ):
        max_num_segments_per_seq *= 2

    segm_output = torch.empty(
        num_seqs,
        num_query_heads,
        max_num_segments_per_seq,
        head_size,
        dtype=torch.float32,
        device=query.device,
    )
    segm_max = torch.empty(
        num_seqs,
        num_query_heads,
        max_num_segments_per_seq,
        dtype=torch.float32,
        device=query.device,
    )
    segm_expsum = torch.empty(
        num_seqs,
        num_query_heads,
        max_num_segments_per_seq,
        dtype=torch.float32,
        device=query.device,
    )

    use_alibi_slopes = alibi_slopes is not None

    if debug_flag and not torch.cuda.is_current_stream_capturing():
        torch.set_printoptions(threshold=10_000)
        print("\nnum_seqs: ", num_seqs)
        print("query shape: ", query.shape)
        print("num query heads: ", num_query_heads)
        print("context_lens: ", context_lens)
        print("block_tables.shape: ", block_tables.shape)
        print("key_cache.shape: ", key_cache.shape)
        print("value_cache.shape: ", value_cache.shape)
        print(block_tables)
        print("query strides: ", query.stride(0), query.stride(1), query.stride(2))
        print("block_tables strides: ", block_tables.stride(0), block_tables.stride(1))
        print(
            "key_cache strides: ",
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            key_cache.stride(4) if len(key_cache.shape)==5 else 1,
        )
        print("output strides: ", output.stride(0), output.stride(1), output.stride(2))
        print(
            "value_cache strides: ",
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
        )
        print("context_lens stride: ", context_lens.stride(0))
        if alibi_slopes is not None:
            print("alibi_slobes stride: ", alibi_slopes.stride(0))

    kernel_paged_attention_3d[(num_seqs, num_query_heads, max_num_segments_per_seq)](
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_tables,
        context_lens_ptr=context_lens,
        alibi_slopes_ptr=alibi_slopes,
        scale=scale,
        cu_q_len_ptr=cu_q_len,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_tables.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        USE_ALIBI_SLOPES=use_alibi_slopes,
        BLOCKS_PER_SEGMENT=blocks_per_segment,
        MAX_NUM_SEGMENTS_PER_SEQ=max_num_segments_per_seq,
        x=key_cache.shape[4],
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        filter_by_query_len=False,
        query_start_len_ptr=None,
    )

    reduce_segments[(num_seqs, num_query_heads)](
        output_ptr=output,
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        context_lens_ptr=context_lens,
        num_query_heads=num_query_heads,
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        BLOCK_SIZE=block_size,
        BLOCKS_PER_SEGMENT=blocks_per_segment,
        MAX_NUM_SEGMENTS_PER_SEQ=max_num_segments_per_seq,
        HEAD_SIZE=head_size,
        filter_by_query_len=False,
        query_start_len_ptr=None,
    )
