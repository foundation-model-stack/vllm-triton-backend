# SPDX-License-Identifier: Apache-2.0
"""
Fused Attention
===============

based on AMDs implementation: https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_flash_attention.py
(occasionally synchronized, last state: https://github.com/rasmith/vllm/commit/71f89c5595a6d1c0b9ce29b68344959cb69aa21a).

Key differences:
- change the wrapper to be imported by vllm-triton backend
- change the autotuner, use triton-dejavu
- minor bug fixes (typos)

Currently only the forward kernel is supported, and contains these features:


1) Fwd with causal masking
2) Arbitrary Q and KV sequence lengths
3) Arbitrary head sizes
4) Multi and grouped query attention
5) Variable sequence lengths
6) ALiBi and matrix bias

Not currently supported:

1) Non power of two head dims

"""

import sys
import os
import torch
from typing import NamedTuple

import triton
import triton.language as tl
import triton_dejavu

from ..utils.triton_utils import unpack_grid


SUPPORTED_LAYOUTS = ["thd", "bhsd", "bshd"]


class MetaData:
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlens_q = 0
    max_seqlens_k = 0
    bias = None
    alibi_slopes = None
    causal = False
    persistent = None
    num_contexts = 0
    varlen = False
    layout = None
    dropout_p, return_encoded_softmax = 0.0, False

    def __init__(self, sm_scale=1.0):
        self.sm_scale = sm_scale

    def set_varlen_params(
        self, cu_seqlens_q, cu_seqlens_k, max_seqlens_q=None, max_seqlens_k=None
    ):
        self.varlen = True
        self.layout = "thd"
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        # Without "varlen", there should still be one sequence.
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        # NOTE(ngl): to avoid   "RuntimeError: CUDA error: operation not permitted when
        #   stream is capturing" /  "RuntimeError: Offset increment outside graph capture
        #   encountered unexpectedly."
        if max_seqlens_q is not None:
            self.max_seqlens_q = max_seqlens_q
        else:
            for i in range(0, self.num_contexts):
                self.max_seqlens_q = max(
                    cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item(),
                    self.max_seqlens_q,
                )
        if max_seqlens_k is not None:
            self.max_seqlens_k = max_seqlens_k
        else:
            for i in range(0, self.num_contexts):
                self.max_seqlens_k = max(
                    cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item(),
                    self.max_seqlens_k,
                )

    def set_persistent(self, persistent):
        self.persistent = persistent

    def need_bias(self, bias, batch, nheads, seqlen_q, seqlen_k):
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.shape[0] == 1
        assert bias.shape[2:] == (seqlen_q, seqlen_k)
        self.bias = bias

    def need_alibi(self, alibi_slopes, batch, nheads):
        assert alibi_slopes.is_cuda
        assert alibi_slopes.dim() == 2
        assert alibi_slopes.shape[0] == batch
        assert alibi_slopes.shape[1] == nheads
        self.alibi_slopes = alibi_slopes

    def need_causal(self):
        self.causal = True

    def need_dropout(self, dropout_p, return_encoded_softmax):
        self.dropout_p = dropout_p
        self.return_encoded_softmax = return_encoded_softmax

    def check_args(self, q, k, v, o):
        assert q.dim() == k.dim() and q.dim() == v.dim()

        batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, self)
        if self.varlen:
            assert q.dim() == 3
            assert self.cu_seqlens_q is not None
            assert self.cu_seqlens_k is not None
            assert self.num_contexts >= 0
            assert len(self.cu_seqlens_q) == len(self.cu_seqlens_k)
            # TODO: Remove once bias is supported with varlen
            assert self.bias is None
            # TODO:Remove once dropout is supported with varlen
            assert self.dropout_p == 0.0
            assert not self.return_encoded_softmax
            if self.max_seqlens_q > 4096:
                # FIXME(ngl): causes RuntimeError: CUDA error: an illegal memory access was encountered
                #  (with triton 3.2.0)
                # for now, we support only batch size of 64 above prompt length of 4096
                # cu_seqlens_q is one entry longer than batch size
                assert len(self.cu_seqlens_q) <= (64 + 1)
        else:
            assert q.dim() == 4
            assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
            assert self.cu_seqlens_q is None and self.cu_seqlens_k is None
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]

        # TODO: Change assert if we support qkl f8 and v f16
        assert q.dtype == k.dtype and q.dtype == v.dtype
        assert head_size <= 256
        assert o.shape == q.shape
        assert (nheads_q % nheads_k) == 0
        assert self.layout is not None
        assert self.layout == "thd" or not self.varlen


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


@triton.jit
def dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]


@triton.jit
def dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_offsets = dropout_offsets(
        philox_seed, philox_offset, dropout_p, m, n, stride
    ).to(tl.uint32)
    # TODO: use tl.randint for better performance
    return tl.rand(philox_seed, rng_offsets)


@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep


# Convenience function to load with optional boundary checks.
# "First" is the major dim, "second" is the minor dim.
@triton.jit
def load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & (
            offset_second[None, :] < boundary_second
        )
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor


@triton.jit
def print_gpu(prefix, val=None):
    if (tl.program_id(0) == 0) and (
        (tl.program_id(1) == 0) and (tl.program_id(2) == 0)
    ):
        if val is not None:
            tl.device_print(prefix, val)
        else:
            tl.device_print(prefix)


@triton.jit
def compute_alibi_block(
    alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n, transpose=False
):
    # when seqlen_k and seqlen_q are different we want the diagonal to stick to
    # the bottom right of the attention matrix
    # for casual mask we want something like this where
    # (1 is kept and 0 is masked)
    # seqlen_q = 2 and seqlen_k = 5
    #   1 1 1 1 0
    #   1 1 1 1 1
    # seqlen_q = 5 and seqlen_k = 2
    #        0 0
    #        0 0
    #        0 0
    #        1 0
    #        1 1
    # for alibi the diagonal is 0 indicating no penalty for attending to
    # that spot and increasing penalty for attending further from the diagonal
    # e.g. alibi_slope = 1, seqlen_q = 2, seqlen_k = 5, offs_m = [0, 1, 2, 3],
    # offs_n = [0, 1, 2, 3, 4], transpose = False
    # 1. offs_m[:,None] = [[0],
    #                       [1],
    # 2. offs_m[:,None] + seqlen_k = [[5],
    #                                  [6],
    # 3. offs_m[:,None] + seqlen_k - seqlen_q = [[3],
    #                                             [4],
    # 4. offs_m[:,None] + seqlen_k - seqlen_q - offs_n[None,:] =
    #    [[3], - [[0, 1, 2, 3, 4]] =  [[ 3, 2, 1, 0,-1],
    #    [4],                           [ 4, 3, 2, 1, 0]]
    # 5. -1 * alibi_slope * tl.abs(relative_pos_block) = [[ -3, -2, -1, 0,-1],
    #                                                     [ -4, -3, -2, -1, 0]],
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


def compute_alibi_tensor(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device="cuda").unsqueeze(
        -1
    )  # (N_CTX_Q, 1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32, device="cuda").unsqueeze(
        0
    )  # (1, N_CTX_K)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q - k_idx)  # (N_CTX_Q, N_CTX_K)
    return (
        -1 * alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos
    )  # (Z, H, N_CTX_Q, N_CTX_K)


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_ptrs,
    v_ptrs,
    bias_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    start_m,
    actual_seqlen_k,
    actual_seqlen_q,
    dropout_p,
    philox_seed,
    batch_philox_offset,
    encoded_sm_ptrs,
    block_min,
    block_max,
    offs_n_causal,
    masked_blocks,
    n_extra_tokens,
    alibi_slope,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    QK_SCALE: tl.constexpr,
):
    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        k_offs_n = start_n + tl.arange(0, BLOCK_N) if MASK_STEPS else None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
        k = load_fn(k_ptrs, k_offs_k, k_offs_n, ACTUAL_BLOCK_DMODEL, actual_seqlen_k)
        if PRE_LOAD_V:
            # We can use the same offsets as k, just with dims transposed.
            v = load_fn(
                v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL
            )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        if MASK_STEPS:  # noqa SIM102
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if
            # not is_modulo_mn.
            # last step might get wasted but that is okay.
            # Check if this masking works for that case.
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
        # -- compute qk ----
        qk += tl.dot(q, k) * QK_SCALE

        if bias_ptrs is not None:
            bias_offs_n = start_n + tl.arange(0, BLOCK_N) if MASK_STEPS else None
            bias = load_fn(
                bias_ptrs, OFFS_M, bias_offs_n, actual_seqlen_q, actual_seqlen_k
            )
            # While bias is added after multiplying qk with sm_scale,
            # our optimization to use 2^x instead of e^x results in an
            # additional scale factor of log2(e) which we must also multiply
            # the bias with.
            qk += bias * 1.44269504089

        if alibi_slope is not None:
            # Compute the global position of each token within the sequence
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(
                alibi_slope,
                actual_seqlen_q,
                actual_seqlen_k,
                global_m_positions,
                global_n_positions,
            )
            qk += alibi_block * 1.44269504089  # scale factor of log2(e)

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            philox_offset = (
                batch_philox_offset
                + start_m * BLOCK_M * actual_seqlen_k
                + start_n
                - BLOCK_N
            )
            keep = dropout_mask(
                philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, actual_seqlen_k
            )
            if RETURN_ENCODED_SOFTMAX:
                tl.store(
                    encoded_sm_ptrs,
                    tl.where(keep, p, -p).to(encoded_sm_ptrs.type.element_ty),
                )
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            tl.store(encoded_sm_ptrs, p.to(encoded_sm_ptrs.type.element_ty))
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(
                v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL
            )
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij

        acc += tl.dot(p.to(v.type.element_ty), v)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if bias_ptrs is not None:
            bias_ptrs += BLOCK_N * stride_bn
        if RETURN_ENCODED_SOFTMAX:
            encoded_sm_ptrs += BLOCK_N
    return acc, l_i, m_i


old_amd_configs = [
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 64, "PRE_LOAD_V": False}, num_stages=1, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "PRE_LOAD_V": False}, num_stages=1, num_warps=4
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "PRE_LOAD_V": False}, num_stages=1, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "PRE_LOAD_V": True}, num_stages=1, num_warps=4
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "PRE_LOAD_V": False}, num_stages=1, num_warps=4
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "PRE_LOAD_V": False}, num_stages=1, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "PRE_LOAD_V": False}, num_stages=1, num_warps=8
    ),
    # TODO: This config fails with head_size not pow2 with data mismatches. Check why.
    #    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1, num_warps=4),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 16, "PRE_LOAD_V": False}, num_stages=1, num_warps=4
    ),
]


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in (
        "gfx940",
        "gfx941",
        "gfx942",
        "gfx90a",
        "gfx908",
    )


def is_rdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in (
        "gfx1030",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1200",
        "gfx1201",
    )


def get_cdna_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "waves_per_eu": 2,
                "PRE_LOAD_V": False,
                "GRID_CU_MULTIP": 2,
            },
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "waves_per_eu": 2,
                "PRE_LOAD_V": False,
                "GRID_CU_MULTIP": 2,
            },
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "waves_per_eu": 3,
                "PRE_LOAD_V": False,
                "GRID_CU_MULTIP": 2,
            },
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "waves_per_eu": 1,
                "PRE_LOAD_V": False,
                "GRID_CU_MULTIP": 2,
            },
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 32,
                "waves_per_eu": 2,
                "PRE_LOAD_V": False,
                "GRID_CU_MULTIP": 2,
            },
            num_stages=1,
            num_warps=4,
        ),
    ], [
        "IS_CAUSAL",
        "dropout_p",
        "MAX_SEQLENS_Q",
        "MAX_SEQLENS_K",
        "ACTUAL_BLOCK_DMODEL",
        "VARLEN",
        "HQ",
        "HK",
    ]


def get_rdna_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 32,
                "waves_per_eu": 4,
                "PRE_LOAD_V": False,
                "GRID_CU_MULTIP": 2,
            },
            num_stages=1,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 32,
                "waves_per_eu": 2,
                "PRE_LOAD_V": False,
                "GRID_CU_MULTIP": 2,
            },
            num_stages=1,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 16,
                "waves_per_eu": 4,
                "PRE_LOAD_V": False,
                "GRID_CU_MULTIP": 2,
            },
            num_stages=1,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 16,
                "waves_per_eu": 2,
                "PRE_LOAD_V": False,
                "GRID_CU_MULTIP": 2,
            },
            num_stages=1,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 16,
                "BLOCK_N": 16,
                "waves_per_eu": 4,
                "PRE_LOAD_V": False,
                "GRID_CU_MULTIP": 2,
            },
            num_stages=1,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 16,
                "BLOCK_N": 16,
                "waves_per_eu": 2,
                "PRE_LOAD_V": False,
                "GRID_CU_MULTIP": 2,
            },
            num_stages=1,
            num_warps=2,
        ),
        # Fall-back config.
        triton.Config(
            {
                "BLOCK_M": 16,
                "BLOCK_N": 16,
                "waves_per_eu": 1,
                "PRE_LOAD_V": False,
                "GRID_CU_MULTIP": 2,
            },
            num_stages=1,
            num_warps=2,
        ),
    ], [
        "IS_CAUSAL",
        "dropout_p",
        "MAX_SEQLENS_Q",
        "MAX_SEQLENS_K",
        "ACTUAL_BLOCK_DMODEL",
        "VARLEN",
        "HQ",
        "HK",
    ]


def get_autotune_configs():
    if is_rdna():
        return get_rdna_autotune_configs()
    elif is_cdna():
        return get_cdna_autotune_configs()
    else:
        raise ValueError("Unknown Device Type")


# if is_hip():
#   autotune_configs, autotune_keys = get_autotune_configs()

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
    total_tokens, num_query_heads, head_size = args["Q"].shape
    _, num_kv_heads, head_size = args["K"].shape
    dtype_size = args["Q"].element_size()
    num_seqs = args["cu_seqlens_q"].shape[0]

    num_bytes = (
        (dtype_size * total_tokens * num_query_heads * head_size)
        + (dtype_size * total_tokens * num_kv_heads * head_size * 2)
        + num_seqs * dtype_size  # dtype size? not ptr size?
    )
    num_flops = total_tokens * num_kv_heads * head_size * 7  # TODO?
    return {
        "name": f"triton_zrl_flash_attention_2_<grid:{grid_x}x{grid_y}x{grid_z}>_<cluster:{cluster_x}x{cluster_y}x{cluster_z}>_<warps:{num_warps}>_<shared:{shared_memory}>_<stages:{num_stages}>",
        "flops16": num_flops,
        "bytes": num_bytes,
    }


# not as lambda, for python3.9
def fallback_heuristic(key):
    # TODO: use key?
    ret = triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 64, "PRE_LOAD_V": False, "GRID_CU_MULTIP": 2},
        num_stages=1,
        num_warps=8,
    )
    return ret


def informed_fallback_next(key, cache):
    # key[27] = MAX_SEQLENS_Q
    ret = cache[min(cache.keys(), key=lambda x: abs(x - key[27]))]
    return ret


def informed_fallback_previous(key, cache):
    # key[27] = MAX_SEQLENS_Q
    sorted_keys = sorted(cache.keys())
    next_idx = sorted_keys.index(min(sorted_keys, key=lambda x: abs(x - key[27])))
    prev_key = sorted_keys[max(0, next_idx - 1)]
    ret = cache[prev_key]
    return ret


def prepare_informed_fallback(cache):
    # key[27] = MAX_SEQLENS_Q
    ret = {int(k[27]): c for k, c in cache.items()}
    return ret


# lazy functions for paper evals
use_bo = lambda: os.getenv("NGL_EXP_USE_BO", "0") == "1"
use_random = lambda: os.getenv("NGL_EXP_USE_RANDOM_SEARCH", "0") == "1"
bo_time = lambda: int(os.getenv("NGL_EXP_BO_TIME", "360"))


def _select_informed_fallback():
    fallback_mode = os.getenv("NGL_EXP_FALLBACK", "none")
    if fallback_mode == "static":
        return None, None
    if fallback_mode == "next":
        return informed_fallback_next, prepare_informed_fallback
    if fallback_mode == "previous":
        return informed_fallback_previous, prepare_informed_fallback
    return informed_fallback_next, prepare_informed_fallback


# defaults to work without env
select_fallback_heuristic = lambda: (
    fallback_heuristic if os.getenv("NGL_EXP_FALLBACK", "none") == "static" else None
)
select_informed_fallback = lambda: _select_informed_fallback()[0]
select_prepare_informed_fallback = lambda: _select_informed_fallback()[1]


@triton_dejavu.autotune(
    config_space=triton_dejavu.ConfigSpace(
        {
            "BLOCK_M": [16, 32, 64, 128, 256],
            "BLOCK_N": [16, 32, 64, 128, 256],
            "PRE_LOAD_V": [True, False],
            "GRID_CU_MULTIP": [2],  # only relevant for persistent
        },
        kwarg_conditions=[
            lambda kwarg: kwarg["BLOCK_M"] >= kwarg["BLOCK_N"],
            lambda kwarg: kwarg["BLOCK_M"] != 64
            or "H100"
            not in gpu_name,  # BLOCK_M not 64! still causes segfault on H100 (also with latest triton)
            # lambda kwarg: kwarg['BLOCK_N'] != 128 or kwarg['PRE_LOAD_V'] == False,  # if BLOCK_N = 128 and pre_load there are numerical errors on H100
            # lambda kwarg: kwarg['PRE_LOAD_V'] == True or 'H100' in gpu_name,   # to ensure pre load for all other GPUs
            lambda kwarg: kwarg["BLOCK_N"] >= 32 or "H100" not in gpu_name,
        ],
        num_warps=[2, 4, 8],
        num_stages=[1, 2, 4, 6, 8],
        num_ctas=[1],  # although supported by H100, it causes a segfault if >1
        # TODO: add warp specialization parameters
    ),
    key=[
        "HQ",
        "HK",
        "IS_CAUSAL",
        "dropout_p",
        "BLOCK_DMODEL",
        "stride_qz",
        "stride_qh",
        "stride_qm",
        "stride_qk",
        "stride_kz",
        "stride_kh",
        "stride_kn",
        "stride_kk",
        "stride_vz",
        "stride_vh",
        "stride_vn",
        "stride_vk",
        "stride_oz",
        "stride_oh",
        "stride_om",
        "stride_on",
        "stride_bz",
        "stride_bh",
        "stride_bm",
        "stride_bn",
        "stride_az",
        "stride_ah",
        "MAX_SEQLENS_Q",
        "MAX_SEQLENS_K",
        "VARLEN",
        "ACTUAL_BLOCK_DMODEL",
        # 'BATCH_SIZE'
    ],
    # verbose=False,
    use_cuda_graph=True,
    # informed_fallback = lambda key, cache: cache[min(cache.keys(), key=lambda x: abs(x - key[27]))],
    # prepare_informed_fallback = lambda cache: {int(k[27]): c for k, c in cache.items()},
    # informed_fallback = informed_fallback_next,
    # prepare_informed_fallback = prepare_informed_fallback,
    # fallback_heuristic=select_fallback_heuristic(),
    fallback_heuristic=fallback_heuristic,
    informed_fallback=select_informed_fallback(),
    prepare_informed_fallback=select_prepare_informed_fallback(),
    # use_bo=True,
    use_bo=use_bo(),
    use_random_search=use_random(),
    search_max_search_t=bo_time(),
    # bo_max_search_t=20,  # for debugging...
    # bo_max_search_t=180,
    # bo_max_search_t=360,
    # bo_max_search_t=420,
    # bo_max_search_t=720,
    # bo_max_search_t=1200,
    # bo_max_search_t=2400,
    # bo_max_search_t=5800,
    # bo_max_share=0.2,  # will anyhow timeout...
    search_max_share=1.0,  # will anyhow timeout...
    # bo_max_share=0.05,
    # bo_max_repeat=3,
    # bo_max_repeat=1,
    custom_data_storage=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "dejavu_data")
    ),
)
@triton.jit(launch_metadata=metadata_fn)
def attn_fwd(
    Q,
    K,
    V,
    bias,
    SM_SCALE: tl.constexpr,
    L,
    Out,
    # treat all autotuner keys as constants?
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    stride_bz: tl.constexpr,
    stride_bh: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_az: tl.constexpr,
    stride_ah: tl.constexpr,
    cu_seqlens_q,
    cu_seqlens_k,
    dropout_p,
    philox_seed,
    PERSISTENT: tl.constexpr,
    PERSISTENT_DYNAMIC: tl.constexpr,
    atomic_counter,
    NUM_CU: tl.constexpr,
    B: tl.constexpr,
    philox_offset_base,
    encoded_softmax,
    alibi_slopes,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    VARLEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    USE_BIAS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    # autotuner args last!
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    GRID_CU_MULTIP: tl.constexpr,
):

    if PERSISTENT:  # if persistent, kernel loops over multiple tiles
        NUM_WG = NUM_CU * GRID_CU_MULTIP  # number of workgroups launched
        num_tiles_per_head = tl.cdiv(
            MAX_SEQLENS_Q, BLOCK_M
        )  # the number of work units (tiles) of a single head
        num_tiles_per_sample = num_tiles_per_head * HQ  # times the num of heads
        num_tiles_total = num_tiles_per_sample * B  # times the num of samples
        if PERSISTENT_DYNAMIC:
            tile_id = atomic_counter.atomic_add(
                1
            )  # returns the value BEFORE the atomic operation
        else:
            tile_id = tl.program_id(0)
    else:  # standard, kernel processes only one tile
        tile_id = 0
        num_tiles_total = 1

    while tile_id < num_tiles_total:  # loops more than once only if PERSISTENT
        if PERSISTENT:
            # tile id basically tells us the Q block we are handling
            # which batch sample
            off_z = tile_id // num_tiles_per_sample
            # which head are we inside the sample
            off_h_q = tile_id % num_tiles_per_sample // num_tiles_per_head
            # at which tile are we inside the head
            start_m = tile_id % num_tiles_per_sample % num_tiles_per_head
        else:
            start_m = tl.program_id(0)
            off_h_q = tl.program_id(1)
            off_z = tl.program_id(2)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        # as we can't have return statements inside while loop in Triton
        continue_condition = True

        if VARLEN:
            cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
            cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
            seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
            # We have a one-size-fits-all grid in id(0).
            # Some seqlens might be too small for all start_m so for those
            # we return early.
            if start_m * BLOCK_M > seqlen_q:
                continue_condition = False
                # return
            cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
            cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        else:
            cu_seqlens_q_start = 0
            cu_seqlens_k_start = 0
            seqlen_q = MAX_SEQLENS_Q
            seqlen_k = MAX_SEQLENS_K

        if continue_condition:
            # Now we compute whether we need to exit early due to causal
            # masking.
            # This is because for seqlen_q > seqlen_k, M rows of the attn
            # scores are completely masked, resulting in 0s written to the
            # output, and inf written to LSE. We don't need to do any GEMMs
            # in this case.
            # This block of code determines what N is, and if this WG is
            # operating on those M rows.
            n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
            if IS_CAUSAL:
                # If seqlen_q == seqlen_k, the attn scores are a square matrix.
                # If seqlen_q != seqlen_k, attn scores are rectangular which
                # means the causal mask boundary is bottom right aligned, and
                # ends at either the top edge (seqlen_q < seqlen_k) or left
                # edge.
                # This captures the decrease in n_blocks if we have a
                # rectangular attn matrix
                n_blocks_seqlen = cdiv_fn(
                    (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N
                )
                # This is what adjusts the block_max for the current WG, only
                # if IS_CAUSAL. Otherwise we want to always iterate through
                # all n_blocks
                n_blocks = min(n_blocks, n_blocks_seqlen)
                # If we have no blocks after adjusting for seqlen deltas,
                # this WG is part of the blocks that are all 0. We exit early.
                if n_blocks <= 0:
                    o_offset = (
                        Out
                        + off_z * stride_oz
                        + off_h_q * stride_oh
                        + cu_seqlens_q_start * stride_om
                    )
                    o_ptrs = (
                        o_offset
                        + offs_m[:, None] * stride_om
                        + offs_d[None, :] * stride_on
                    )
                    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=Out.type.element_ty)
                    o_ptrs_mask = (offs_m[:, None] < seqlen_q).broadcast_to(
                        [BLOCK_M, BLOCK_DMODEL]
                    )
                    # We still need to write 0s to the result
                    tl.store(o_ptrs, acc, mask=o_ptrs_mask)
                    # The tensor allocated for L is based on MAX_SEQLENS_Q as
                    # that is statically known.
                    l_ptrs = (
                        L
                        + off_z * HQ * MAX_SEQLENS_Q
                        + off_h_q * MAX_SEQLENS_Q
                        + offs_m
                    )
                    # We store inf to LSE, not -inf because in the bwd pass,
                    # we subtract this from qk which makes it -inf, such that
                    # exp(qk - inf) = 0 for these masked blocks.
                    l_value = tl.full([BLOCK_M], value=float("inf"), dtype=tl.float32)
                    l_ptrs_mask = offs_m < MAX_SEQLENS_Q
                    tl.store(l_ptrs, l_value, mask=l_ptrs_mask)
                    # TODO: Should dropout and return encoded softmax be
                    # handled here too?
                    continue_condition = False
                    # return

            if continue_condition:
                # If MQA / GQA, set the K and V head offsets appropriately.
                GROUP_SIZE: tl.constexpr = HQ // HK
                off_h_k = off_h_q // GROUP_SIZE if GROUP_SIZE != 1 else off_h_q

                n_extra_tokens = 0
                if seqlen_k < BLOCK_N:
                    n_extra_tokens = BLOCK_N - seqlen_k
                elif seqlen_k % BLOCK_N:
                    n_extra_tokens = seqlen_k % BLOCK_N
                PADDED_HEAD: tl.constexpr = ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL

                # Compute pointers for all the tensors used in this kernel.
                q_offset = (
                    Q
                    + off_z * stride_qz
                    + off_h_q * stride_qh
                    + cu_seqlens_q_start * stride_qm
                )
                q_ptrs = (
                    q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
                )
                k_offset = (
                    K
                    + off_z * stride_kz
                    + off_h_k * stride_kh
                    + cu_seqlens_k_start * stride_kn
                )
                k_ptrs = (
                    k_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
                )
                v_offset = (
                    V
                    + off_z * stride_vz
                    + off_h_k * stride_vh
                    + cu_seqlens_k_start * stride_vk
                )
                v_ptrs = (
                    v_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
                )
                # Compute pointers for all the scale tensors used in this
                # kernel.

                if USE_BIAS:
                    # Note: this might get large enough to overflow on
                    # some configs
                    bias_offset = off_h_q * stride_bh
                    bias_ptrs = (
                        bias
                        + bias_offset
                        + offs_m[:, None] * stride_bm
                        + offs_n[None, :] * stride_bn
                    )
                else:
                    bias_ptrs = None

                if USE_ALIBI:
                    a_offset = off_z * stride_az + off_h_q * stride_ah
                    alibi_slope = tl.load(alibi_slopes + a_offset)
                else:
                    alibi_slope = None

                if ENABLE_DROPOUT:
                    off_hz = off_z * HQ + off_h_q
                    batch_philox_offset = (
                        philox_offset_base + off_hz * seqlen_q * seqlen_k
                    )
                else:
                    batch_philox_offset = 0
                # We can ask to return the dropout mask without actually doing
                # any dropout. In this case, we return an invalid pointer so
                # indicate the mask is not valid.
                if RETURN_ENCODED_SOFTMAX:
                    encoded_sm_base = encoded_softmax + off_h_q * seqlen_q * seqlen_k
                    encoded_sm_ptrs = (
                        encoded_sm_base + offs_m[:, None] * seqlen_k + offs_n[None, :]
                    )
                else:
                    encoded_sm_ptrs = None
                # initialize pointer to m and l
                m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
                l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
                acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
                # scale sm_scale by log_2(e) and use 2^x in the loop as we
                # do not have native e^x support in HW.
                QK_SCALE: tl.constexpr = SM_SCALE * 1.44269504089
                # Q is loaded once at the beginning and shared by all N blocks.
                q_ptrs_mask = offs_m[:, None] < seqlen_q
                if PADDED_HEAD:
                    q_ptrs_mask = q_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
                q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)

                # Here we compute how many full and masked blocks we have.
                padded_block_k = n_extra_tokens != 0
                is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
                if IS_CAUSAL:
                    # There are always at least BLOCK_M // BLOCK_N masked
                    # blocks. Additionally there might be one more due to
                    # dissimilar seqlens.
                    masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
                else:
                    # Padding on Q does not need to be masked in the FA loop.
                    masked_blocks = padded_block_k
                # if IS_CAUSAL, not is_modulo_mn does not always result in an
                # additional block. In this case we might exceed n_blocks so
                # pick the min.
                masked_blocks = min(masked_blocks, n_blocks)
                n_full_blocks = n_blocks - masked_blocks
                block_min = 0
                block_max = n_blocks * BLOCK_N
                # Compute for full blocks. Here we set causal to false
                # regardless of its actual value because there is no masking.
                # Similarly we do not need padding.
                if n_full_blocks > 0:
                    block_max = (n_blocks - masked_blocks) * BLOCK_N
                    acc, l_i, m_i = _attn_fwd_inner(
                        acc,
                        l_i,
                        m_i,
                        q,
                        k_ptrs,
                        v_ptrs,
                        bias_ptrs,
                        stride_kn,
                        stride_vk,
                        stride_bn,
                        start_m,
                        seqlen_k,
                        seqlen_q,
                        dropout_p,
                        philox_seed,
                        batch_philox_offset,
                        encoded_sm_ptrs,
                        # _, _, offs_n_causal, masked_blocks, n_extra_tokens, _
                        block_min,
                        block_max,
                        0,
                        0,
                        0,
                        alibi_slope,
                        # IS_CAUSAL, ....
                        False,
                        BLOCK_M,
                        BLOCK_DMODEL,
                        BLOCK_N,
                        offs_m,
                        offs_n,
                        # _, MASK_STEPS, ...
                        PRE_LOAD_V,
                        False,
                        ENABLE_DROPOUT,
                        RETURN_ENCODED_SOFTMAX,
                        PADDED_HEAD,
                        ACTUAL_BLOCK_DMODEL,
                        QK_SCALE,
                    )

                    block_min = block_max
                    block_max = n_blocks * BLOCK_N

                tl.debug_barrier()
                # Remaining blocks, if any, are full / not masked.
                if masked_blocks > 0:
                    if IS_CAUSAL:
                        offs_n_causal = offs_n + (seqlen_q - seqlen_k)
                    else:
                        offs_n_causal = 0
                    k_ptrs += n_full_blocks * BLOCK_N * stride_kn
                    v_ptrs += n_full_blocks * BLOCK_N * stride_vk
                    if USE_BIAS:
                        bias_ptrs += n_full_blocks * BLOCK_N * stride_bn
                    if RETURN_ENCODED_SOFTMAX:
                        encoded_sm_ptrs += n_full_blocks * BLOCK_N
                    acc, l_i, m_i = _attn_fwd_inner(
                        acc,
                        l_i,
                        m_i,
                        q,
                        k_ptrs,
                        v_ptrs,
                        bias_ptrs,
                        stride_kn,
                        stride_vk,
                        stride_bn,
                        start_m,
                        seqlen_k,
                        seqlen_q,
                        dropout_p,
                        philox_seed,
                        batch_philox_offset,
                        encoded_sm_ptrs,
                        block_min,
                        block_max,
                        offs_n_causal,
                        masked_blocks,
                        n_extra_tokens,
                        alibi_slope,
                        IS_CAUSAL,
                        BLOCK_M,
                        BLOCK_DMODEL,
                        BLOCK_N,
                        offs_m,
                        offs_n,
                        # _, MASK_STEPS, ...
                        PRE_LOAD_V,
                        True,
                        ENABLE_DROPOUT,
                        RETURN_ENCODED_SOFTMAX,
                        PADDED_HEAD,
                        ACTUAL_BLOCK_DMODEL,
                        QK_SCALE,
                    )

                # epilogue
                # This helps the compiler do Newton Raphson on l_i vs on acc
                # which is much larger.
                l_recip = 1 / l_i[:, None]
                acc = acc * l_recip

                if ENABLE_DROPOUT:
                    acc = acc / (1 - dropout_p)
                # If seqlen_q > seqlen_k but the delta is not a multiple of
                # BLOCK_M, then we have one block with a row of all NaNs which
                # come from computing softmax over a row of all
                # -infs (-inf - inf = NaN). We check for that here and store
                # 0s where there are NaNs as these rows should've been zeroed
                # out.
                end_m_idx = (start_m + 1) * BLOCK_M
                start_m_idx = start_m * BLOCK_M
                causal_start_idx = seqlen_q - seqlen_k
                acc = acc.to(Out.type.element_ty)
                if IS_CAUSAL:  # noqa: SIM102
                    if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
                        out_mask_boundary = tl.full(
                            (BLOCK_DMODEL,), causal_start_idx, dtype=tl.int32
                        )
                        mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
                        out_ptrs_mask = (
                            mask_m_offsets[:, None] >= out_mask_boundary[None, :]
                        )
                        z = 0.0
                        acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))
                # write back LSE
                l_ptrs = (
                    L + off_z * HQ * MAX_SEQLENS_Q + off_h_q * MAX_SEQLENS_Q + offs_m
                )
                # If seqlen_q not multiple of BLOCK_M, we need to mask out the
                # last few rows. This is only true for the last M block. For
                # others, overflow_size will be -ve
                overflow_size = end_m_idx - seqlen_q
                if overflow_size > 0:
                    boundary = tl.full(
                        (BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32
                    )
                    l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
                    tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=l_ptrs_mask)
                else:
                    tl.store(l_ptrs, m_i + tl.math.log2(l_i))

                # write back O
                o_offset = (
                    Out
                    + off_z * stride_oz
                    + off_h_q * stride_oh
                    + cu_seqlens_q_start * stride_om
                )
                o_ptrs = (
                    o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
                )
                o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL], 1, dtype=tl.int1)
                if overflow_size > 0:
                    o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
                if PADDED_HEAD:
                    o_ptrs_mask = o_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
                tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)

        if PERSISTENT:
            if PERSISTENT_DYNAMIC:
                tile_id = atomic_counter.atomic_add(1)
            else:
                tile_id += NUM_WG
        else:
            tile_id = num_tiles_total  # break after single tile


def get_shape_from_layout(q, k, metadata):
    assert metadata.layout in SUPPORTED_LAYOUTS, "Got unsupported layout."
    if metadata.layout == "thd":
        nheads_q, nheads_k = q.shape[1], k.shape[1]
        head_size = q.shape[-1]
        batch = metadata.num_contexts
    elif metadata.layout == "bhsd":
        batch, nheads_q, _, head_size = q.shape
        nheads_k = k.shape[1]
    elif metadata.layout == "bshd":
        batch, _, nheads_q, head_size = q.shape
        nheads_k = k.shape[2]
    return batch, nheads_q, nheads_k, head_size


# TODO: This can probably optimized to have fewer lines of code.
def get_strides_from_layout(q, k, v, o, metadata):
    assert metadata.layout in SUPPORTED_LAYOUTS, "Got unsupported layout."
    if metadata.layout == "thd":
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    elif metadata.layout == "bhsd":
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
        o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
    elif metadata.layout == "bshd":
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
    return q_strides, k_strides, v_strides, o_strides


def triton_wrapper_forward_prefill(
    q,
    k,
    v,
    max_seqlen_q,
    max_seqlen_k,
    cu_seqlens_q,
    cu_seqlens_k,
    causal=False,
    sm_scale=1.0,
    bias=None,
    config=None,
    in_place_output=None,
    do_not_return_softmax_encodings=True,
):
    metadata = MetaData()
    metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    metadata.causal = causal
    metadata.sm_scale = sm_scale
    metadata.bias = bias

    if cu_seqlens_q is not None:
        metadata.num_contexts = len(cu_seqlens_q) - 1
    if in_place_output is None:
        o = torch.empty_like(q, dtype=q.dtype)
    else:
        o = in_place_output
    if metadata.bias is not None:
        assert metadata.bias.numel() < 2**31

    if config is None:
        config = {}

    # triton implementation
    metadata.check_args(q, k, v, o)

    batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, metadata)
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(
        q, k, v, o, metadata
    )

    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)

    # encoded_softmax is used to validate dropout behavior vs the PyTorch
    # SDPA math backend reference.  We zero this out to give a consistent
    # starting point and then populate it with the output of softmax with
    # the sign bit set according to the dropout mask. The resulting return
    # allows this mask to be fed into the reference implementation for
    # testing only.  This return holds no useful output aside from
    # debugging.
    if metadata.return_encoded_softmax:
        encoded_softmax = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2], k.shape[2]),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        encoded_softmax = None

    M = torch.empty(
        (batch, nheads_q, metadata.max_seqlens_q),
        device=q.device,
        dtype=torch.float32,
    )

    # Seed the RNG so we get reproducible results for testing.
    philox_seed = 0x1BF52
    philox_offset = 0x1D4B42

    if metadata.bias is not None:
        bias_strides = (
            metadata.bias.stride(0),
            metadata.bias.stride(1),
            metadata.bias.stride(2),
            metadata.bias.stride(3),
        )
    else:
        bias_strides = (0, 0, 0, 0)

    if metadata.alibi_slopes is not None:
        alibi_strides = (
            metadata.alibi_slopes.stride(0),
            metadata.alibi_slopes.stride(1),
        )
    else:
        alibi_strides = (0, 0)

    # number of compute units available
    NUM_CU = torch.cuda.get_device_properties("cuda").multi_processor_count

    # TODO: test persistent
    if metadata.persistent is not None:
        grid = lambda META: (
            min(
                NUM_CU * META["GRID_CU_MULTIP"],
                triton.cdiv(metadata.max_seqlens_q, META["BLOCK_M"]) * nheads_q * batch,
            ),
        )
    else:
        grid = lambda META: (
            triton.cdiv(metadata.max_seqlens_q, META["BLOCK_M"]),
            nheads_q,
            batch,
        )

    atomic_counter = torch.zeros([1], device=q.device, dtype=torch.int32)

    # if debug_flag and not torch.cuda.is_current_stream_capturing():
    #     torch.set_printoptions(threshold=10_000)
    #     print(f"grid {grid}, BLOCK_DMODEL {padded_d_model}")

    attn_fwd[grid](
        q,
        k,
        v,
        metadata.bias,
        metadata.sm_scale,
        M,
        o,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        *bias_strides,
        *alibi_strides,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        dropout_p=metadata.dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset,
        encoded_softmax=encoded_softmax,
        alibi_slopes=metadata.alibi_slopes,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL=head_size,
        MAX_SEQLENS_Q=metadata.max_seqlens_q,
        MAX_SEQLENS_K=metadata.max_seqlens_k,
        IS_CAUSAL=metadata.causal,
        VARLEN=metadata.varlen,
        BLOCK_DMODEL=padded_d_model,
        USE_BIAS=bool(metadata.bias),
        USE_ALIBI=bool(metadata.alibi_slopes),
        ENABLE_DROPOUT=metadata.dropout_p > 0.0,
        RETURN_ENCODED_SOFTMAX=metadata.return_encoded_softmax,
        PERSISTENT=metadata.persistent is not None,
        PERSISTENT_DYNAMIC=metadata.persistent == "dynamic",
        NUM_CU=NUM_CU,
        atomic_counter=atomic_counter,
        B=batch,
        **config,
    )

    if do_not_return_softmax_encodings:
        return o
    return o, encoded_softmax
