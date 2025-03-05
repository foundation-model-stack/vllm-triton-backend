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
    k_scale,  # float32
    v_scale,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.constexpr,  # int, should be equal to max_num_blocks_per_seq
    query_stride_0: tl.constexpr,  # int
    query_stride_1: tl.constexpr,  # int, should be equal to head_size
    output_stride_0: tl.constexpr,  # int
    output_stride_1: tl.constexpr,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    x: tl.constexpr,  # int
    stride_k_cache_0: tl.constexpr,  # int
    stride_k_cache_1: tl.constexpr,  # int
    stride_k_cache_2: tl.constexpr,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_k_cache_4: tl.constexpr,  # int
    stride_v_cache_0: tl.constexpr,  # int
    stride_v_cache_1: tl.constexpr,  # int
    stride_v_cache_2: tl.constexpr,  # int
    stride_v_cache_3: tl.constexpr,  # int
    filter_by_query_len: tl.constexpr,  # bool
    query_start_len_ptr,  # [num_seqs+1]
):
    seq_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    kv_head_idx = query_head_idx // num_queries_per_kv

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    query_offset = (
        cur_batch_in_all_start_index * query_stride_0 + query_head_idx * query_stride_1
    )

    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # Q : (HEAD_SIZE,)
    Q = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED),
        mask=dim_mask,
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    m = tl.full([1], float("-inf"), dtype=tl.float32)
    l = tl.full([1], 1.0, dtype=tl.float32)
    acc = tl.zeros([HEAD_SIZE_PADDED], dtype=tl.float32)

    # context len for this particualr sequence
    context_len = tl.load(context_lens_ptr + seq_idx)

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_head_idx)

    num_blocks = cdiv_fn(context_len, BLOCK_SIZE)

    # iterate through tiles
    for j in range(0, num_blocks):

        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        offs_n = tl.arange(0, BLOCK_SIZE)
        offs_d = tl.arange(0, HEAD_SIZE)

        v_offset = (
            physical_block_idx * stride_v_cache_0
            + kv_head_idx * stride_v_cache_1
            + offs_d[:, None] * stride_v_cache_2
            + offs_n[None, :] * stride_v_cache_3
        )

        k_offset = (
            physical_block_idx * stride_k_cache_0
            + kv_head_idx * stride_k_cache_1
            + (offs_d[:, None] // x) * stride_k_cache_2
            + offs_n[None, :] * stride_k_cache_3
            + (offs_d[:, None] % x) * stride_k_cache_4
        )

        # K : (HEAD_SIZE, BLOCK_SIZE)
        K_load = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None], other=0.0)

        if K_load.dtype.is_fp8():
            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (HEAD_SIZE, BLOCK_SIZE)
        V_load = tl.load(value_cache_ptr + v_offset, mask=dim_mask[:, None], other=0.0)

        if V_load.dtype.is_fp8():
            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

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

    output_offset = (
        cur_batch_in_all_start_index * output_stride_0
        + query_head_idx * output_stride_1
    )

    tl.store(
        output_ptr + output_offset + tl.arange(0, HEAD_SIZE_PADDED), acc, mask=dim_mask
    )


def paged_attention_triton_2d(
    output,
    query,
    key_cache,
    value_cache,
    scale,
    k_scale,
    v_scale,
    kv_cache_dtype,
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

    # Conversion of FP8 Tensor from uint8 storage to
    # appropriate torch.dtype for interpretation by Triton
    if "fp8" in kv_cache_dtype:
        assert key_cache.dtype == torch.uint8
        assert value_cache.dtype == torch.uint8

        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            target_dtype = torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2":
            target_dtype = torch.float8_e5m2
        else:
            raise ValueError("Unsupported FP8 dtype:", kv_cache_dtype)

        key_cache = key_cache.view(target_dtype)
        value_cache = value_cache.view(target_dtype)

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
            key_cache.stride(4) if len(key_cache.shape) == 5 else 1,
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
        k_scale=k_scale,
        v_scale=v_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_tables.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=use_alibi_slopes,
        x=key_cache.shape[4] if len(key_cache.shape) == 5 else 1,
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4) if len(key_cache.shape) == 5 else 1,
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        filter_by_query_len=False,
        query_start_len_ptr=None,
    )
