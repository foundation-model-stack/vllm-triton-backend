import os
import sys
import pytest
import random
from enum import Enum
from typing import List, Optional, Tuple, Union
import torch
from vllm.platforms import current_platform
from vllm.utils import get_max_shared_memory_bytes, get_kv_cache_torch_dtype
from vllm_utils import (
    create_kv_caches_with_random,
    ref_single_query_cached_kv_attention,
    ref_multi_query_kv_attention,
)
import triton
import triton.language as tl
import math


MY_IUT = [
    e for e in os.environ.get("MY_IUT", "").split(",") if len(e) > 0
]  # my implementations under test (IUT)


""" These values are taken from the vLLM unit tests;
    Let's try to keep them fixed to that so we are
    confident that kernels meet minimum requirements.
"""
FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512

NUM_BLOCKS = 4321  # Arbitrary values for testing
PARTITION_SIZE = 512
DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_GEN_SEQS = [7]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
NUM_HEADS = [
    (40, 40),
    (64, 8),
]  # Arbitrary values for testing, order:  num_query_heads, num_kv_heads
HEAD_SIZES = [64, 80, 120, 256]
BLOCK_SIZES = [16, 32]
USE_ALIBI = [False, True]
KV_CACHE_DTYPE = ["auto", "fp8"]
SEEDS = [0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]
CAUSAL_FLASH = [True]  # vLLM only needs causal=True

# Test range of scales for Q/K/V distribution
# If None defaults to head_size**-0.5
MAX_VALUES = [None]


class Implementation(Enum):
    VLLM_CUDA_V1 = 0
    VLLM_CUDA_V2 = 1
    TRITON_2D = 2
    BASELINE_TRITON = 3
    XFORMERS = 4
    FLASH_ATTN = 5
    FLASHINFER = 6
    TRITON_FP8 = 7
    TRITON_3D = 8


IMPLEMENTATION_UT = [
    Implementation.TRITON_2D,
    Implementation.TRITON_3D,
    Implementation.BASELINE_TRITON,
    Implementation.VLLM_CUDA_V1,
    Implementation.VLLM_CUDA_V2,
    Implementation.XFORMERS,
    Implementation.FLASH_ATTN,
    Implementation.FLASHINFER,
    Implementation.TRITON_FP8,
]

impl_translate = {i.name: i.value for i in Implementation}

if len(MY_IUT) > 0:
    IMPLEMENTATION_UT = []
    for ci_value in MY_IUT:
        IMPLEMENTATION_UT.append(Implementation(impl_translate[ci_value]))
    print(f"Modified test setup:\n\tIMPLEMENATION_UT: {IMPLEMENTATION_UT}")


@pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("use_alibi", USE_ALIBI)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("implementation", IMPLEMENTATION_UT)
@pytest.mark.parametrize("max_value", MAX_VALUES)
@torch.inference_mode()
def test_decoding_attention(
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
    device: str,
    implementation: Implementation,
    max_value: Optional[float],
) -> None:
    if max_value is None:
        max_value = head_size**-0.5

    # V100 does not support bf16
    if torch.cuda.get_device_capability()[0] < 8 and dtype is torch.bfloat16:
        pytest.skip()

    if (
        implementation == Implementation.TRITON_2D
        or implementation == Implementation.TRITON_3D
    ):
        if (not math.log(head_size, 2).is_integer()) or kv_cache_dtype == "fp8":
            pytest.skip()
    elif implementation in [Implementation.VLLM_CUDA_V1, Implementation.VLLM_CUDA_V2]:
        if kv_cache_dtype == "fp8" and head_size % 16:
            pytest.skip()
    elif implementation == Implementation.BASELINE_TRITON:
        if not math.log(MAX_SEQ_LEN, 2).is_integer():
            pytest.skip()
    elif implementation == Implementation.XFORMERS:
        if (
            not math.log(head_size, 2).is_integer()
            or num_heads[0] != num_heads[1]
            or kv_cache_dtype == "fp8"
            or use_alibi
            or dtype is torch.float32
        ):
            pytest.skip()
    elif implementation == Implementation.FLASH_ATTN:
        if head_size % 32 != 0 or dtype is torch.float32 or kv_cache_dtype == "fp8":
            pytest.skip()
    elif implementation == Implementation.TRITON_FP8:
        if (
            not math.log(head_size, 2).is_integer()
            or use_alibi
            or dtype in [torch.bfloat16, torch.float32]
            or kv_cache_dtype == "fp8"
        ):
            pytest.skip()
    elif implementation == Implementation.FLASHINFER:
        # note: FlashInfer does support fp8 KV cache; need to resolve
        if (
            head_size not in [64, 128, 256]
            or use_alibi
            or dtype is torch.float32
            or kv_cache_dtype == "fp8"
        ):
            pytest.skip()

    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    # needed for triton: https://github.com/triton-lang/triton/issues/2925#issuecomment-1890251449
    torch.cuda.set_device(device)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    query.uniform_(-max_value, max_value)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads

    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=torch.float)

    if implementation == Implementation.TRITON_FP8:
        # will be fixed, but for now this kernel need constant seq len within batch
        seq_lens = [MAX_SEQ_LEN for _ in range(num_seqs)]
    else:
        seq_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
        seq_lens[-1] = MAX_SEQ_LEN

    max_seq_len = max(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int)

    # Create the block tables.
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables_lst: List[List[int]] = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables_lst.append(block_table)

    block_tables = torch.tensor(block_tables_lst, dtype=torch.int)

    use_alignment_optimization = False
    if implementation in [Implementation.VLLM_CUDA_V1, Implementation.VLLM_CUDA_V2]:
        use_alignment_optimization = True

    # Create the KV caches.
    key_caches, value_caches = create_kv_caches_with_random(
        NUM_BLOCKS,
        block_size,
        1,
        num_kv_heads,
        head_size,
        max_value,
        kv_cache_dtype,
        dtype,
        seed,
        device,
        alignment_optimization=use_alignment_optimization,
    )

    key_cache, value_cache = key_caches[0], value_caches[0]

    if implementation == Implementation.BASELINE_TRITON:
        from callers import BaselineTritonCaller as Caller
    elif implementation == Implementation.TRITON_2D:
        from callers import Triton2dAttentionDecodeCaller as Caller
    elif implementation == Implementation.TRITON_3D:
        from callers import Triton3dAttentionDecodeCaller as Caller
    elif implementation == Implementation.VLLM_CUDA_V1:
        from callers import VllmCudaV1Caller as Caller
    elif implementation == Implementation.VLLM_CUDA_V2:
        from callers import VllmCudaV2Caller as Caller
    elif implementation == Implementation.XFORMERS:
        from callers import XformersCaller as Caller
    elif implementation == Implementation.FLASH_ATTN:
        from callers import FlashAttnDecodeCaller as Caller
    elif implementation == Implementation.TRITON_FP8:
        from callers import TritonFp8Caller as Caller
    elif implementation == Implementation.FLASHINFER:
        from callers import FlashInferCaller as Caller

    if Caller.requires_allocated_output:
        output = torch.empty_like(query)
    else:
        output = None

    output_ = Caller.make_call_func(
        output,
        query,
        key_cache,
        value_cache,
        num_seqs,
        seq_lens,
        max_seq_len,
        scale,
        block_tables,
        alibi_slopes,
        kv_cache_dtype,
    )()

    output = Caller.select_output(output, output_)

    # Run the reference implementation.
    if kv_cache_dtype == "fp8":
        # Convert cache data back to dtype.
        x = 16 // torch.tensor([], dtype=dtype).element_size()
        key_cache_shape = (NUM_BLOCKS, num_kv_heads, head_size // x, block_size, x)
        dequantized_key_cache = torch.empty(
            size=key_cache_shape, dtype=dtype, device=device
        )
        ops.convert_fp8(dequantized_key_cache, key_cache)
        key_cache = dequantized_key_cache

        value_cache_shape = value_cache.shape
        dequantized_value_cache = torch.empty(
            size=value_cache_shape, dtype=dtype, device=device
        )
        ops.convert_fp8(dequantized_value_cache, value_cache)
        value_cache = dequantized_value_cache

    ref_output = torch.empty_like(query)
    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        num_queries_per_kv,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        alibi_slopes,
    )

    atol, rtol = 1e-3, 1e-5
    if kv_cache_dtype == "fp8":
        atol, rtol = 1e-2, 1e-5

    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)


# TODO(woosuk): Add tests for USE_ALIBI=True.
@pytest.mark.parametrize("num_seqs", NUM_PREFILL_SEQS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("causal", CAUSAL_FLASH)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("implementation", IMPLEMENTATION_UT)
@pytest.mark.parametrize("max_value", MAX_VALUES)
@torch.inference_mode()
def test_prefill_attention(
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    causal: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
    implementation: Implementation,
    max_value: Optional[float],
) -> None:
    if torch.cuda.get_device_capability()[0] < 8:
        # reduce operations are not supported (?)
        pytest.skip()

    if implementation == Implementation.TRITON_3D:
        if (not math.log(head_size, 2).is_integer()) or (head_size > 256):
            pytest.skip()
        if MAX_SEQ_LEN > 4096 and num_seqs > 64:
            # FIXME(ngl): causes RuntimeError: CUDA error: an illegal memory access was encountered
            #  (with triton 3.2.0)
            # for now, we support only batch size of 64 above prompt length of 4096
            pytest.skip()
    elif implementation in [Implementation.VLLM_CUDA_V1, Implementation.VLLM_CUDA_V2]:
        pytest.skip()
    elif implementation == Implementation.BASELINE_TRITON:
        pytest.skip()
    elif implementation == Implementation.XFORMERS:
        pytest.skip()
    elif implementation == Implementation.FLASH_ATTN:
        if head_size % 32 != 0 or dtype is torch.float32:
            pytest.skip()
    elif implementation == Implementation.FLASHINFER:
        pytest.skip()

    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    max_len = min(MAX_SEQ_LEN, 4096)
    seq_lens = random.sample(range(1, max_len), num_seqs)
    num_tokens = sum(seq_lens)
    max_seqlen = max(seq_lens)

    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    qkv = torch.empty(
        num_tokens, num_query_heads + 2 * num_kv_heads, head_size, dtype=dtype
    )
    qkv.uniform_(-max_value, max_value)
    query, key, value = qkv.split([num_query_heads, num_kv_heads, num_kv_heads], dim=1)

    num_queries_per_kv = num_query_heads // num_kv_heads
    if num_queries_per_kv > 1:
        # Handle MQA and GQA
        key = torch.repeat_interleave(key, num_queries_per_kv, dim=1)
        value = torch.repeat_interleave(value, num_queries_per_kv, dim=1)

    if implementation == Implementation.FLASH_ATTN:
        from callers import FlashAttnPrefillCaller as Caller
    elif implementation == Implementation.TRITON_3D:
        from callers import Triton3dAttentionPrefillCaller as Caller

    if Caller.requires_allocated_output:
        output = torch.empty_like(query)
    else:
        output = None

    cu_seq_lens = [0]
    for seq_len in seq_lens:
        cu_seq_lens.append(cu_seq_lens[-1] + seq_len)
    prompt_lens_tensor = torch.tensor(cu_seq_lens, dtype=torch.int32, device=device)

    call_func_under_test = Caller.make_call_func(
        output,
        query,
        key,
        value,
        prompt_lens_tensor,
        prompt_lens_tensor,
        max_seqlen,
        max_seqlen,
        scale,
        causal,
    )

    output_ = call_func_under_test()
    output = Caller.select_output(output, output_)

    # reference implementation
    ref_output = ref_multi_query_kv_attention(
        cu_seq_lens,
        query,
        key,
        value,
        scale,
        dtype,
    )

    # TODO: keep for AMD
    # atol = get_default_atol(output) if current_platform.is_rocm() else 1e-3
    # rtol = get_default_rtol(output) if current_platform.is_rocm() else 1e-5
    atol = 1e-3
    rtol = 1e-5
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)


if __name__ == "__main__":
    # Get arguments to pass to pytest
    if len(sys.argv) >= 1:
        args = [__file__]
        filter_args = ""
        for ca in sys.argv[1:]:
            if ca[0] == "-":
                args.append(ca)
            else:
                filter_args += f"{ca} or "
        if len(filter_args) > 2:
            args.append(f"-k {filter_args[:-3]}")
        # print(f"starting pytest with args: {args}")
        rc = pytest.main(args=args)
    else:
        rc = pytest.main(args=[__file__])

    raise SystemExit(rc)
