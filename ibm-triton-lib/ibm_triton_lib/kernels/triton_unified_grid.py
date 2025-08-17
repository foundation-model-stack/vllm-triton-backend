# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

import triton_dejavu
import os

logger = init_logger(__name__)


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(boundary_ptr, target_idx, num_seqs):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(boundary_ptr + mid)
        if val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton_dejavu.autotune(
    config_space=triton_dejavu.ConfigSpace(
        {
            "BLOCK_M": [16, 32, 64, 128, 256, 512],
            "TILE_SIZE": [16, 32, 64, 128, 256, 512],
        },
        num_warps=[2, 4, 8],
        num_stages=[1, 2, 4, 6, 8],
        num_consumer_groups=[0, 2, 4, 8],
        num_buffers_warp_spec=[0, 3, 6, 9],
        # num_consumer_groups=[2, 4],
        # num_buffers_warp_spec=[3, 6],
        conditions=[
            # ensure consistency for ws
            lambda c: (c.num_consumer_groups != 0 and c.num_buffers_warp_spec != 0) \
                or (c.num_consumer_groups == 0 and c.num_buffers_warp_spec == 0),
        ]
    ),
    # this list is longer, since it would be used for multiple models
    key=[
        "num_query_heads",
        "num_queries_per_kv",
        "BLOCK_SIZE",
        "HEAD_SIZE",
        "HEAD_SIZE_PADDED",
        "SLIDING_WINDOW",
        "stride_k_cache_3",
        "stride_v_cache_3",
        "is_prefill",
    ],
    custom_data_storage=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "dejavu_data")
    ),
    use_cuda_graph=True,
    # use_bo=True,
    # search_max_search_t=360,
    # search_max_search_t=720,
    use_random_search=True,
    search_max_search_t=1800,
    # informed_fallback=informed_fallback_next,
    # prepare_informed_fallback=prepare_informed_fallback,
    # fallback_heuristic=fallback_heuristic_dt2,
    ignore_dtypes=True,
)
@triton.heuristics(
       {"BLOCK_Q": lambda args: args['BLOCK_M'] // args['num_queries_per_kv']},
)
@triton.jit
def kernel_unified_attention_2d(
        output_ptr,  # [num_tokens, num_query_heads, head_size]
        query_ptr,  # [num_tokens, num_query_heads, head_size]
        key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
        value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
        block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
        seq_lens_ptr,  # [num_seqs]
        alibi_slopes_ptr,  # [num_query_heads]
        qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
        scale,  # float32
        k_scale,  # float32
        v_scale,  # float32
        softcap,  # float32
        num_query_heads: tl.constexpr,  # int
        num_queries_per_kv: tl.constexpr,  # int
        block_table_stride: tl.int64,  # int
        query_stride_0: tl.int64,  # int
        query_stride_1: tl.int64,  # int, should be equal to head_size
        output_stride_0: tl.int64,  # int
        output_stride_1: tl.int64,  # int, should be equal to head_size
        qq_bias_stride_0: tl.int64,  # int
        BLOCK_SIZE: tl.constexpr,  # int
        HEAD_SIZE: tl.constexpr,  # int
        HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
        USE_ALIBI_SLOPES: tl.constexpr,  # bool
        USE_QQ_BIAS: tl.constexpr,  # bool
        USE_SOFTCAP: tl.constexpr,  # bool
        SLIDING_WINDOW: tl.constexpr,  # int
        stride_k_cache_0: tl.int64,  # int
        stride_k_cache_1: tl.int64,  # int
        stride_k_cache_2: tl.int64,  # int
        stride_k_cache_3: tl.constexpr,  # int
        stride_v_cache_0: tl.int64,  # int
        stride_v_cache_1: tl.int64,  # int
        stride_v_cache_2: tl.int64,  # int
        stride_v_cache_3: tl.constexpr,  # int
        query_start_len_ptr,  # [num_seqs+1]
        num_seqs: tl.int32,
        seq_idx_offset,  # int
        block_q_seq_boundaries_ptr, # [num_prefills] or None
        is_prefill: tl.constexpr,
        max_q_block_idx: tl.int32,  # int
        q_block_iterations: tl.int32,  # int
        TILE_SIZE: tl.constexpr,  # int must be power of 2
        BLOCK_Q: tl.constexpr,  # int
        BLOCK_M: tl.constexpr,  # int
):
    if tl.program_id(0) * q_block_iterations > max_q_block_idx:
        return

    for q_block_global_idx in range(tl.program_id(0) * q_block_iterations, min((tl.program_id(0) + 1) * q_block_iterations, max_q_block_idx + 1)):
        kv_head_idx = tl.program_id(1)
    
        if is_prefill:
            seq_idx = find_seq_idx(block_q_seq_boundaries_ptr, q_block_global_idx, num_seqs)
            q_block_start_idx = tl.load(block_q_seq_boundaries_ptr + seq_idx)
        else:
            seq_idx = q_block_global_idx
            q_block_start_idx = seq_idx
        seq_idx = seq_idx + seq_idx_offset
    
        q_block_local_idx = q_block_global_idx - q_block_start_idx
    
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    
        cur_batch_query_len = cur_batch_in_all_stop_index \
            - cur_batch_in_all_start_index
    
        #if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        #    return
    
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_SIZE_PADDED)
        offs_t = tl.arange(0, TILE_SIZE)
        query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    
        query_offset_0 = cur_batch_in_all_start_index + query_pos
        query_offset_1 = kv_head_idx * num_queries_per_kv + \
            offs_m % num_queries_per_kv
        query_offset = (query_offset_0[:, None] * query_stride_0 +
                        query_offset_1[:, None] * query_stride_1 + offs_d[None, :])
    
        dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
        query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
        query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)
    
        # Q : (BLOCK_M, HEAD_SIZE_PADDED)
        Q = tl.load(
            query_ptr + query_offset,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            other=0.0,
        )
    
        block_table_offset = seq_idx * block_table_stride
    
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    
        # sequence len for this particular sequence
        seq_len = tl.load(seq_lens_ptr + seq_idx)
    
        # context length for this particular sequences
        context_len = seq_len - cur_batch_query_len
    
        # alibi slope for this head
        if USE_ALIBI_SLOPES:
            alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1,
                                  mask=query_mask_1,
                                  other=0.0)
    
        # query-query attention bias
        if USE_QQ_BIAS:
            qq_bias_row_ptrs = (qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
                                )  # shape: [BLOCK_M]
    
        # compute the length of the longest sequence prefix spanned by any
        # query token in the current q_block (q_block_local_idx)
        max_seq_prefix_len = context_len + q_block_local_idx * BLOCK_Q + (
            BLOCK_M - 1) // num_queries_per_kv + 1
    
        # adjust for potential padding in the last q_block by considering the
        # actual sequence length
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    
        # calculate the number of tiles that need to be processed to
        # cover the longest sequence prefix (due to causal masking, tiles beyond
        # this prefix can be skipped)
        num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)
    
        # iterate through tiles
        for j in range(0, num_tiles):
            seq_offset = j * TILE_SIZE + offs_t
            tile_mask = seq_offset < max_seq_prefix_len
    
            physical_block_idx = tl.load(block_tables_ptr + block_table_offset +
                                         seq_offset // BLOCK_SIZE).to(tl.int64)
    
            v_offset = (physical_block_idx[:, None] * stride_v_cache_0 +
                        kv_head_idx * stride_v_cache_2 +
                        offs_d[None, :] * stride_v_cache_3 +
                        (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1)
    
            k_offset = (physical_block_idx[None, :] * stride_k_cache_0 +
                        kv_head_idx * stride_k_cache_2 +
                        offs_d[:, None] * stride_k_cache_3 +
                        (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1)
    
            # K : (HEAD_SIZE, TILE_SIZE)
            K_load = tl.load(key_cache_ptr + k_offset,
                             mask=dim_mask[:, None] & tile_mask[None, :],
                             other=0.0)
    
            if K_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    K = K_load
                else:
                    K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
            else:
                K = K_load
    
            # V : (TILE_SIZE, HEAD_SIZE)
            V_load = tl.load(value_cache_ptr + v_offset,
                             mask=dim_mask[None, :] & tile_mask[:, None],
                             other=0.0)
    
            if V_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    V = V_load
                else:
                    V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
            else:
                V = V_load
    
            seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1
    
            # S : (BLOCK_M, TILE_SIZE)
            S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
    
            S += scale * tl.dot(Q, K)
    
            if USE_SOFTCAP:
                S = apply_softcap(S, softcap)
    
            S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
                         S, float("-inf"))
    
            if SLIDING_WINDOW > 0:
                S = tl.where((context_len + query_pos[:, None] - seq_offset)
                             < SLIDING_WINDOW, S, float("-inf"))
    
            if USE_ALIBI_SLOPES:
                S += alibi_slope[:, None] * (seq_offset - context_len)
    
            if USE_QQ_BIAS:
                # compute key positions relative to query section
                key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
                # load bias only for keys that correspond to queries
                is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
                qq_bias = tl.load(
                    qq_bias_row_ptrs + key_rel_pos[None, :],
                    mask=is_query_key[None, :],  # avoid OOB for context keys
                    other=0.0,
                )
                S += qq_bias
    
            # compute running maximum
            # m_j : (BLOCK_M,)
            m_j = tl.maximum(M, tl.max(S, axis=1))
    
            # For sliding window there's a chance the max is -inf due to masking of
            # the entire row. In this case we need to set m_j 0 to avoid NaN
            m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    
            # P : (BLOCK_M, TILE_SIZE)
            P = tl.exp(S - m_j[:, None])
    
            # l_j : (BLOCK_M,)
            l_j = tl.sum(P, axis=1)
    
            # alpha : (BLOCK_M, )
            alpha = tl.exp(M - m_j)
    
            # acc : (BLOCK_M, HEAD_SIZE_PADDED)
            acc = acc * alpha[:, None]
    
            # update constants
            L = L * alpha + l_j
            M = m_j
    
            # acc : (BLOCK_M, HEAD_SIZE_PADDED)
            acc += tl.dot(P.to(V.dtype), V)
    
        # epilogue
        acc = acc / L[:, None]
    
        output_offset = (query_offset_0[:, None] * output_stride_0 +
                         query_offset_1[:, None] * output_stride_1 +
                         offs_d[None, :])
    
        tl.store(
            output_ptr + output_offset,
            acc,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )


@triton_dejavu.autotune(
    config_space=triton_dejavu.ConfigSpace(
        {
            "BLOCK_M": [16, 32, 64, 128, 256, 512],
            "TILE_SIZE": [16, 32, 64, 128, 256, 512],
        },
        num_warps=[2, 4, 8],
        num_stages=[1, 2, 4, 6, 8],
        num_consumer_groups=[0, 2, 4, 8],
        num_buffers_warp_spec=[0, 3, 6, 9],
        # num_consumer_groups=[2, 4],
        # num_buffers_warp_spec=[3, 6],
        conditions=[
            # ensure consistency for ws
            lambda c: (c.num_consumer_groups != 0 and c.num_buffers_warp_spec != 0) \
                or (c.num_consumer_groups == 0 and c.num_buffers_warp_spec == 0),
        ]
    ),
    # this list is longer, since it would be used for multiple models
    key=[
        "num_query_heads",
        "num_queries_per_kv",
        "BLOCK_SIZE",
        "HEAD_SIZE",
        "HEAD_SIZE_PADDED",
        "SLIDING_WINDOW",
        "stride_k_cache_3",
        "stride_v_cache_3",
        "NUM_SEGMENTS_PER_SEQ",
    ],
    custom_data_storage=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "dejavu_data")
    ),
    use_cuda_graph=True,
    # use_bo=True,
    # search_max_search_t=360,
    # search_max_search_t=720,
    use_random_search=True,
    search_max_search_t=1800,
    # informed_fallback=informed_fallback_next,
    # prepare_informed_fallback=prepare_informed_fallback,
    # fallback_heuristic=fallback_heuristic_dt2,
    ignore_dtypes=True,
)
@triton.heuristics(
       {"BLOCK_Q": lambda args: args['BLOCK_M'] // args['num_queries_per_kv']},
)
@triton.jit
def kernel_unified_attention_3d(
        segm_output_ptr,
        # [num_tokens, num_query_heads, num_segments, head_size]
        segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
        segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
        query_ptr,  # [num_tokens, num_query_heads, head_size]
        key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
        value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
        block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
        seq_lens_ptr,  # [num_seqs]
        alibi_slopes_ptr,  # [num_query_heads]
        qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
        scale,  # float32
        k_scale,  # float32
        v_scale,  # float32
        softcap,  # float32
        num_query_heads: tl.constexpr,  # int
        num_queries_per_kv: tl.constexpr,  # int
        block_table_stride: tl.int64,  # int
        query_stride_0: tl.int64,  # int
        query_stride_1: tl.int64,  # int, should be equal to head_size
        qq_bias_stride_0: tl.int64,  # int
        BLOCK_SIZE: tl.constexpr,  # int
        HEAD_SIZE: tl.constexpr,  # int
        HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
        USE_ALIBI_SLOPES: tl.constexpr,  # bool
        USE_QQ_BIAS: tl.constexpr,  # bool
        USE_SOFTCAP: tl.constexpr,  # bool
        SLIDING_WINDOW: tl.constexpr,  # int
        stride_k_cache_0: tl.int64,  # int
        stride_k_cache_1: tl.int64,  # int
        stride_k_cache_2: tl.int64,  # int
        stride_k_cache_3: tl.constexpr,  # int
        stride_v_cache_0: tl.int64,  # int
        stride_v_cache_1: tl.int64,  # int
        stride_v_cache_2: tl.int64,  # int
        stride_v_cache_3: tl.constexpr,  # int
        query_start_len_ptr,  # [num_seqs+1]
        num_seqs: tl.int32,
        NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
        seq_idx_iterations: tl.int32,  # int
        BLOCK_Q: tl.constexpr,  # int
        TILE_SIZE: tl.constexpr,  # int, must be power of 2
        BLOCK_M: tl.constexpr,  # int
):
    if tl.program_id(0) * seq_idx_iterations >= num_seqs:
        return

    for seq_idx in range(tl.program_id(0) * seq_idx_iterations, min((tl.program_id(0) + 1) * seq_idx_iterations, num_seqs)):
        kv_head_idx = tl.program_id(1)
        segm_idx = tl.program_id(2)
    
        # sequence len for this particular sequence
        seq_len = tl.load(seq_lens_ptr + seq_idx)
    
        # number of segments for this particular sequence
        num_segments = NUM_SEGMENTS_PER_SEQ
        tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)
    
        #if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        #    return
    
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_SIZE_PADDED)
        offs_t = tl.arange(0, TILE_SIZE)
        query_pos = offs_m // num_queries_per_kv
    
        query_offset_0 = seq_idx + query_pos #cur_batch_in_all_start_index + query_pos
        query_offset_1 = kv_head_idx * num_queries_per_kv + \
            offs_m % num_queries_per_kv
        query_offset = (query_offset_0[:, None] * query_stride_0 +
                        query_offset_1[:, None] * query_stride_1 + offs_d[None, :])
    
        dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
        query_mask_0 = tl.where(query_pos < 1, 1, 0).to(tl.int1)
        query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)
    
        # Q : (BLOCK_M, HEAD_SIZE_PADDED)
        Q = tl.load(
            query_ptr + query_offset,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            other=0.0,
        )
    
        block_table_offset = seq_idx * block_table_stride
    
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    
        # context length for this particular sequences
        context_len = seq_len - 1
    
        # alibi slope for this head
        if USE_ALIBI_SLOPES:
            alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1,
                                  mask=query_mask_1,
                                  other=0.0)
    
        # query-query attention bias
        if USE_QQ_BIAS:
            qq_bias_row_ptrs = (qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
                                )  # shape: [BLOCK_M]
    
        num_tiles = cdiv_fn(seq_len, TILE_SIZE)
    
        # iterate through tiles within current segment
        for j in range(
                segm_idx * tiles_per_segment,
                min((segm_idx + 1) * tiles_per_segment, num_tiles),
        ):
            seq_offset = j * TILE_SIZE + offs_t
            tile_mask = seq_offset < seq_len
    
            physical_block_idx = tl.load(block_tables_ptr + block_table_offset +
                                         seq_offset // BLOCK_SIZE).to(tl.int64)
    
            v_offset = (physical_block_idx[:, None] * stride_v_cache_0 +
                        kv_head_idx * stride_v_cache_2 +
                        offs_d[None, :] * stride_v_cache_3 +
                        (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1)
    
            k_offset = (physical_block_idx[None, :] * stride_k_cache_0 +
                        kv_head_idx * stride_k_cache_2 +
                        offs_d[:, None] * stride_k_cache_3 +
                        (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1)
    
            # K : (HEAD_SIZE, TILE_SIZE)
            K_load = tl.load(key_cache_ptr + k_offset,
                             mask=dim_mask[:, None] & tile_mask[None, :],
                             other=0.0)
    
            if K_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    K = K_load
                else:
                    K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
            else:
                K = K_load
    
            # V : (TILE_SIZE, HEAD_SIZE)
            V_load = tl.load(value_cache_ptr + v_offset,
                             mask=dim_mask[None, :] & tile_mask[:, None],
                             other=0.0)
    
            if V_load.dtype.is_fp8():
                if Q.dtype.is_fp8():
                    V = V_load
                else:
                    V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
            else:
                V = V_load
    
            seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1
    
            # S : (BLOCK_M, TILE_SIZE)
            S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
            S += scale * tl.dot(Q, K)
    
            if USE_SOFTCAP:
                S = apply_softcap(S, softcap)
    
            S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
                         S, float("-inf"))
    
            if SLIDING_WINDOW > 0:
                S = tl.where((context_len + query_pos[:, None] - seq_offset)
                             < SLIDING_WINDOW, S, float("-inf"))
    
            if USE_ALIBI_SLOPES:
                S += alibi_slope[:, None] * (seq_offset - context_len)
    
            if USE_QQ_BIAS:
                # compute key positions relative to query section
                key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
                # load bias only for keys that correspond to queries
                is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
                qq_bias = tl.load(
                    qq_bias_row_ptrs + key_rel_pos[None, :],
                    mask=is_query_key[None, :],  # avoid OOB for context keys
                    other=0.0,
                )
                S += qq_bias
    
            # compute running maximum
            # m_j : (BLOCK_M,)
            m_j = tl.maximum(M, tl.max(S, axis=1))
    
            # For sliding window there's a chance the max is -inf due to masking of
            # the entire row. In this case we need to set m_j 0 to avoid NaN
            m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    
            # P : (BLOCK_M, TILE_SIZE,)
            P = tl.exp(S - m_j[:, None])
    
            # l_j : (BLOCK_M,)
            l_j = tl.sum(P, axis=1)
    
            # alpha : (BLOCK_M, )
            alpha = tl.exp(M - m_j)
    
            # acc : (BLOCK_M, HEAD_SIZE_PADDED)
            acc = acc * alpha[:, None]
    
            # update constants
            L = L * alpha + l_j
            M = m_j
    
            # acc : (BLOCK_M, HEAD_SIZE_PADDED)
            acc += tl.dot(P.to(V.dtype), V)
    
        segm_output_offset = (
            query_offset_0[:, None].to(tl.int64) *
            (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
            query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
            segm_idx * HEAD_SIZE_PADDED + tl.arange(0, HEAD_SIZE_PADDED)[None, :])
        tl.store(
            segm_output_ptr + segm_output_offset,
            acc,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )
        segm_offset = (query_offset_0.to(tl.int64) *
                       (num_query_heads * NUM_SEGMENTS_PER_SEQ) +
                       query_offset_1 * NUM_SEGMENTS_PER_SEQ + segm_idx)
        tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
        tl.store(segm_expsum_ptr + segm_offset,
                 L,
                 mask=query_mask_0 & query_mask_1)


@triton_dejavu.autotune(
    config_space=triton_dejavu.ConfigSpace(
        {
            "TILE_SIZE": [16, 32, 64, 128, 256, 512],
        },
        num_warps=[2, 4, 8],
        num_stages=[1, 2, 4, 6, 8],
        num_consumer_groups=[0, 2, 4, 8],
        num_buffers_warp_spec=[0, 3, 6, 9],
        # num_consumer_groups=[2, 4],
        # num_buffers_warp_spec=[3, 6],
        conditions=[
            # ensure consistency for ws
            lambda c: (c.num_consumer_groups != 0 and c.num_buffers_warp_spec != 0) \
                or (c.num_consumer_groups == 0 and c.num_buffers_warp_spec == 0),
        ]
    ),
    # this list is longer, since it would be used for multiple models
    key=[
        "num_query_heads",
        "HEAD_SIZE",
        "HEAD_SIZE_PADDED",
        "NUM_SEGMENTS_PER_SEQ",
    ],
    custom_data_storage=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "dejavu_data")
    ),
    use_cuda_graph=True,
    # use_bo=True,
    # search_max_search_t=360,
    # search_max_search_t=720,
    use_random_search=True,
    search_max_search_t=1800,
    # informed_fallback=informed_fallback_next,
    # prepare_informed_fallback=prepare_informed_fallback,
    # fallback_heuristic=fallback_heuristic_dt2,
    ignore_dtypes=True,
)
@triton.jit
def reduce_segments(
        output_ptr,  # [num_tokens, num_query_heads, head_size]
        segm_output_ptr,
        #[num_tokens, num_query_heads, max_num_segments, head_size]
        segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
        segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
        seq_lens_ptr,  # [num_seqs]
        num_seqs,  # int
        num_query_heads: tl.constexpr,  # int
        output_stride_0: tl.int64,  # int
        output_stride_1: tl.int64,  # int, should be equal to head_size
        block_table_stride: tl.int64,  # int
        HEAD_SIZE: tl.constexpr,  # int, must be power of 2
        HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
        query_start_len_ptr,  # [num_seqs+1]
        NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
        seq_idx_iterations: tl.int32,  # int
        TILE_SIZE: tl.constexpr,  # int
):
    if tl.program_id(0) * seq_idx_iterations >= num_seqs:
        return

    for seq_idx in range(tl.program_id(0) * seq_idx_iterations, min((tl.program_id(0) + 1) * seq_idx_iterations, num_seqs)):
        query_head_idx = tl.program_id(1)
    
        # sequence len for this particular sequence
        seq_len = tl.load(seq_lens_ptr + seq_idx)
    
        # number of segments for this particular sequence
        num_segments = NUM_SEGMENTS_PER_SEQ
        tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)
    
        # create masks for subsequent loads
        act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
        segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
            [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32)
        dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1,
                            0).to(tl.int1)
    
        # load segment maxima
        segm_offset = (seq_idx.to(tl.int64) *
                       (num_query_heads * NUM_SEGMENTS_PER_SEQ) +
                       query_head_idx * NUM_SEGMENTS_PER_SEQ +
                       tl.arange(0, NUM_SEGMENTS_PER_SEQ))
        segm_max = tl.load(segm_max_ptr + segm_offset,
                           mask=segm_mask,
                           other=float("-inf"))
        overall_max = tl.max(segm_max)
    
        # load and rescale segment exp sums
        segm_expsum = tl.load(segm_expsum_ptr + segm_offset,
                              mask=segm_mask,
                              other=0.0)
        segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
        overall_expsum = tl.sum(segm_expsum)
    
        # load, rescale, and add segment attention outputs
        segm_output_offset = (
            seq_idx.to(tl.int64) *
            (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
            query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
            tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED +
            tl.arange(0, HEAD_SIZE_PADDED)[None, :])
        segm_output = tl.load(
            segm_output_ptr + segm_output_offset,
            mask=segm_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        segm_output *= tl.exp(segm_max - overall_max)[:, None]
        acc_sum = tl.sum(segm_output, axis=0)
        # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
        acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)
    
        # write result
        output_offset = (seq_idx * output_stride_0 +
                         query_head_idx * output_stride_1 +
                         tl.arange(0, HEAD_SIZE_PADDED))
        tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    num_decodes,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    use_split_kv,
    segm_output,
    segm_max,
    segm_expsum,
    BLOCK_M_PREFILL,
    BLOCK_Q_PREFILL,
    BLOCK_M_DECODE,
    BLOCK_Q_DECODE,
    num_q_blocks,
    block_q_seq_boundaries,
    alibi_slopes=None,
    qq_bias=None,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    block_size = v.shape[1]
    assert q.element_size() >= 2 or block_size >= 32, \
        "Block size must be at least 32 for fp8"

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    TILE_SIZE_PREFILL = 32
    TILE_SIZE_DECODE = 32

    LAUNCH_GRID_DIM0_2D_PREFILL = 32
    LAUNCH_GRID_DIM0_2D_DECODE = 32
    LAUNCH_GRID_DIM0_3D_DECODE = 4
    LAUNCH_GRID_DIM0_3D_REDUCE = 4

    # prefill
    if num_seqs > num_decodes:
        kernel_unified_attention_2d[(
            LAUNCH_GRID_DIM0_2D_PREFILL, #num_q_blocks, 
            num_kv_heads,
        )](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            SLIDING_WINDOW=(1 + window_size[0]),
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            num_seqs=num_seqs - num_decodes,
            seq_idx_offset=num_decodes,
            block_q_seq_boundaries_ptr=block_q_seq_boundaries,
            is_prefill=True,
            max_q_block_idx=num_q_blocks-1,
            q_block_iterations=(num_q_blocks + LAUNCH_GRID_DIM0_2D_PREFILL - 1) // LAUNCH_GRID_DIM0_2D_PREFILL
            # tunable parameters
            # BLOCK_M=BLOCK_M_PREFILL,
            # BLOCK_Q=BLOCK_Q_PREFILL,
            # TILE_SIZE=TILE_SIZE_PREFILL,
        )

    # decode
    if num_decodes > 0:
        # select between 2d and 3d (split-kv) kernels
        if not use_split_kv:
            kernel_unified_attention_2d[(
                LAUNCH_GRID_DIM0_2D_DECODE, #num_decodes,
                num_kv_heads,
            )](
                output_ptr=out,
                query_ptr=q,
                key_cache_ptr=k,
                value_cache_ptr=v,
                block_tables_ptr=block_table,
                seq_lens_ptr=seqused_k,
                alibi_slopes_ptr=alibi_slopes,
                qq_bias_ptr=qq_bias,
                scale=softmax_scale,
                k_scale=k_descale,
                v_scale=v_descale,
                softcap=softcap,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                block_table_stride=block_table.stride(0),
                query_stride_0=q.stride(0),
                query_stride_1=q.stride(1),
                output_stride_0=out.stride(0),
                output_stride_1=out.stride(1),
                qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
                BLOCK_SIZE=block_size,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                USE_ALIBI_SLOPES=use_alibi_slopes,
                USE_QQ_BIAS=use_qq_bias,
                USE_SOFTCAP=(softcap > 0),
                SLIDING_WINDOW=(1 + window_size[0]),
                stride_k_cache_0=k.stride(0),
                stride_k_cache_1=k.stride(1),
                stride_k_cache_2=k.stride(2),
                stride_k_cache_3=k.stride(3),
                stride_v_cache_0=v.stride(0),
                stride_v_cache_1=v.stride(1),
                stride_v_cache_2=v.stride(2),
                stride_v_cache_3=v.stride(3),
                query_start_len_ptr=cu_seqlens_q,
                num_seqs=num_decodes,
                seq_idx_offset=0,
                block_q_seq_boundaries_ptr=None,
                is_prefill=False,
                max_q_block_idx=num_decodes-1,
                q_block_iterations=(num_decodes + LAUNCH_GRID_DIM0_2D_DECODE - 1) // LAUNCH_GRID_DIM0_2D_DECODE
                # tunable parameters
                # BLOCK_M=BLOCK_M_DECODE,
                # BLOCK_Q=BLOCK_Q_DECODE,
                # TILE_SIZE=TILE_SIZE_DECODE,
            )
        else:
            # for initial version, NUM_SEGMENTS = 16 is chosen as a default
            # value that showed good performance in tests
            NUM_SEGMENTS = 16
    
#            segm_output = torch.empty(
#                num_decodes,
#                num_query_heads,
#                NUM_SEGMENTS,
#                triton.next_power_of_2(head_size),
#                dtype=torch.float32,
#                device=q.device,
#            )
#            segm_max = torch.empty(
#                num_decodes,
#                num_query_heads,
#                NUM_SEGMENTS,
#                dtype=torch.float32,
#                device=q.device,
#            )
#            segm_expsum = torch.empty(
#                num_decodes,
#                num_query_heads,
#                NUM_SEGMENTS,
#                dtype=torch.float32,
#                device=q.device,
#            )
    
            kernel_unified_attention_3d[(
                LAUNCH_GRID_DIM0_3D_DECODE, #num_decodes,
                num_kv_heads,
                NUM_SEGMENTS
            )](
                segm_output_ptr=segm_output,
                segm_max_ptr=segm_max,
                segm_expsum_ptr=segm_expsum,
                query_ptr=q,
                key_cache_ptr=k,
                value_cache_ptr=v,
                block_tables_ptr=block_table,
                seq_lens_ptr=seqused_k,
                alibi_slopes_ptr=alibi_slopes,
                qq_bias_ptr=qq_bias,
                scale=softmax_scale,
                k_scale=k_descale,
                v_scale=v_descale,
                softcap=softcap,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                block_table_stride=block_table.stride(0),
                query_stride_0=q.stride(0),
                query_stride_1=q.stride(1),
                qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
                BLOCK_SIZE=block_size,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                USE_ALIBI_SLOPES=use_alibi_slopes,
                USE_QQ_BIAS=use_qq_bias,
                USE_SOFTCAP=(softcap > 0),
                SLIDING_WINDOW=(1 + window_size[0]),
                stride_k_cache_0=k.stride(0),
                stride_k_cache_1=k.stride(1),
                stride_k_cache_2=k.stride(2),
                stride_k_cache_3=k.stride(3),
                stride_v_cache_0=v.stride(0),
                stride_v_cache_1=v.stride(1),
                stride_v_cache_2=v.stride(2),
                stride_v_cache_3=v.stride(3),
                query_start_len_ptr=cu_seqlens_q,
                num_seqs=num_decodes,
                NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
                seq_idx_iterations=(num_decodes + LAUNCH_GRID_DIM0_3D_DECODE - 1) // LAUNCH_GRID_DIM0_3D_DECODE
                # tunable parameters
                # BLOCK_Q=BLOCK_Q_DECODE,
                # BLOCK_M=BLOCK_M_DECODE,
                # TILE_SIZE=TILE_SIZE_DECODE,
            )
            reduce_segments[(
                LAUNCH_GRID_DIM0_3D_REDUCE, #num_decodes,
                num_query_heads
            )](
                output_ptr=out,
                segm_output_ptr=segm_output,
                segm_max_ptr=segm_max,
                segm_expsum_ptr=segm_expsum,
                seq_lens_ptr=seqused_k,
                num_seqs=num_seqs,
                num_query_heads=num_query_heads,
                output_stride_0=out.stride(0),
                output_stride_1=out.stride(1),
                block_table_stride=block_table.stride(0),
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                query_start_len_ptr=cu_seqlens_q,
                NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
                seq_idx_iterations=(num_decodes + LAUNCH_GRID_DIM0_3D_REDUCE - 1) // LAUNCH_GRID_DIM0_3D_REDUCE
                # tunable parameters
                # TILE_SIZE=TILE_SIZE_DECODE,
            )
