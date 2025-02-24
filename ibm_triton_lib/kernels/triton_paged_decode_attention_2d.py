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
        "name": f"triton_zrl_paged_attention_2d_<grid:{grid_x}x{grid_y}x{grid_z}>_<cluster:{cluster_x}x{cluster_y}x{cluster_z}>_<warps:{num_warps}>_<shared:{shared_memory}>_<stages:{num_stages}>",
        "flops16": num_flops,
        "bytes": num_bytes,
    }


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit(launch_metadata=metadata_fn)
def kernel_paged_attention_2d(
    output_ptr,  # [num_seqs, num_query_heads, head_size]
    query_ptr,  # [num_seqs, num_query_heads, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size, block_size]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    scale,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    cache_block_stride: tl.constexpr,  # int
    block_table_stride: tl.constexpr,  # int, should be equal to max_num_blocks_per_seq
    query_stride_0: tl.constexpr,  # int
    query_stride_1: tl.constexpr,  # int, should be equal to head_size
    output_stride_0: tl.constexpr,  # int
    output_stride_1: tl.constexpr,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
):
    seq_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    kv_head_idx = query_head_idx // num_queries_per_kv

    query_offset = seq_idx * query_stride_0 + query_head_idx * query_stride_1

    # Q : (HEAD_SIZE,)
    Q = tl.load(query_ptr + query_offset + tl.arange(0, HEAD_SIZE))

    block_table_offset = seq_idx * block_table_stride

    m = tl.full([1], float("-inf"), dtype=tl.float32)
    l = tl.full([1], 1.0, dtype=tl.float32)
    acc = tl.zeros([HEAD_SIZE], dtype=tl.float32)

    # context len for this particualr sequence
    context_len = tl.load(context_lens_ptr + seq_idx)

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_head_idx)

    num_blocks = cdiv_fn(context_len, BLOCK_SIZE)

    # iterate through tiles
    for j in range(0, num_blocks):
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        start_of_block_offset = (
            physical_block_idx * cache_block_stride
            + kv_head_idx * HEAD_SIZE * BLOCK_SIZE
        )

        kv_offset = (
            tl.arange(0, HEAD_SIZE)[:, None] * BLOCK_SIZE
            + tl.arange(0, BLOCK_SIZE)[None, :]
        )

        # K : (HEAD_SIZE, BLOCK_SIZE)
        K = tl.load(key_cache_ptr + start_of_block_offset + kv_offset)

        # V : (HEAD_SIZE, BLOCK_SIZE)
        V = tl.load(value_cache_ptr + start_of_block_offset + kv_offset)

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

    # epilogue
    acc = acc / l

    output_offset = seq_idx * output_stride_0 + query_head_idx * output_stride_1

    tl.store(output_ptr + output_offset + tl.arange(0, HEAD_SIZE), acc)


def paged_attention_triton_2d(
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
):
    use_alibi_slopes = alibi_slopes is not None

    if len(key_cache.shape) == 5 and key_cache.shape[4] != 1:
        raise RuntimeError("5d kv cache not supported")

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

    kernel_paged_attention_2d[
        (
            num_seqs,
            num_query_heads,
        )
    ](
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_tables,
        context_lens_ptr=context_lens,
        alibi_slopes_ptr=alibi_slopes,
        scale=scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        cache_block_stride=key_cache.stride(0),
        block_table_stride=block_tables.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        USE_ALIBI_SLOPES=use_alibi_slopes,
    )
