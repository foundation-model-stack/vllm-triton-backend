# based on https://github.ibm.com/TPA/triton-paged-attention/blob/main/test_working.py

from typing import List, Optional, Tuple, Union, NamedTuple

import torch
import triton
import triton.language as tl

from ibm_triton_lib.utils.triton_utils import unpack_grid


gpu_name = torch.cuda.get_device_name()


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
    num_blocks, num_kv_heads, _, block_size = args["key_cache_ptr"].shape
    _, max_num_blocks_per_seq = args["block_tables_ptr"].shape
    # num tokens are treated as batch
    dtype_size = args["query_ptr"].element_size()
    _, max_context_len, _, _ = args["scratchpad_key_ptr"].shape

    num_bytes = (
        (dtype_size * num_seqs * num_query_heads * head_size)
        + (dtype_size * num_blocks * num_kv_heads * head_size * block_size * 2)
        + num_seqs * max_num_blocks_per_seq * dtype_size  # dtype size? not ptr size?
        + (
            num_seqs * max_context_len * num_query_heads * head_size * dtype_size * 2
        )  # scratchpad
    )
    num_flops = num_blocks * num_kv_heads * head_size * block_size * 7  # TODO?
    return {
        "name": f"triton_vedantroy_paged_attention_1_<grid:{grid_x}x{grid_y}x{grid_z}>_<cluster:{cluster_x}x{cluster_y}x{cluster_z}>_<warps:{num_warps}>_<shared:{shared_memory}>_<stages:{num_stages}>",
        "flops16": num_flops,
        "bytes": num_bytes,
    }


@triton.jit(launch_metadata=metadata_fn)
def paged_attention_v1(
    # need these b/c we can't use view/reshape
    scratchpad_key_ptr,  # [num_seqs, max_context_len, num_heads, head_size]
    scratchpad_value_ptr,  # [num_seqs, max_context_len, num_heads, head_size]
    output_ptr,  # [num_seqs, num_query_heads, head_size]
    query_ptr,  # [num_seqs, num_query_heads, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size, block_size]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lens_ptr,  # [num_seqs]
    scale,  # float32
    num_seqs,  # int
    num_heads,  # int
    cache_block_stride,  # int
    MAX_CONTEXT_LEN: tl.constexpr,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    MAX_NUM_BLOCKS_PER_SEQ: tl.constexpr,  # int, must be power of 2
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    query_offset = seq_idx * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE
    query_head = tl.load(query_ptr + query_offset + tl.arange(0, HEAD_SIZE))
    block_table_offset = seq_idx * MAX_NUM_BLOCKS_PER_SEQ
    context_len = tl.load(context_lens_ptr + seq_idx)

    for tok_idx in range(0, context_len):
        logical_block_idx = tok_idx // BLOCK_SIZE
        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + logical_block_idx
        )

        start_of_block_offset = (
            physical_block_idx * cache_block_stride + head_idx * HEAD_SIZE * BLOCK_SIZE
        )
        tok_idx_within_block = tok_idx % BLOCK_SIZE
        tok_offsets = (
            start_of_block_offset
            + BLOCK_SIZE * tl.arange(0, HEAD_SIZE)
            + tok_idx_within_block
        )

        tok_key = tl.load(key_cache_ptr + tok_offsets)
        tok_value = tl.load(value_cache_ptr + tok_offsets)

        scratchpad_offset = (
            seq_idx * (MAX_CONTEXT_LEN * num_heads * HEAD_SIZE)
            + tok_idx * (num_heads * HEAD_SIZE)
            + head_idx * HEAD_SIZE
        )
        tl.store(
            scratchpad_key_ptr + scratchpad_offset + tl.arange(0, HEAD_SIZE), tok_key
        )
        tl.store(
            scratchpad_value_ptr + scratchpad_offset + tl.arange(0, HEAD_SIZE),
            tok_value,
        )

    # TODO: Not sure if this is necessary
    tl.debug_barrier()

    # scratchpad_key_ptr,  # [num_seqs, max_context_len, num_heads, head_size]
    start_seq_offset = (MAX_CONTEXT_LEN * num_heads * HEAD_SIZE) * seq_idx
    start_tok_offset = (
        start_seq_offset
        + tl.arange(0, MAX_CONTEXT_LEN) * (num_heads * HEAD_SIZE)
        + head_idx * HEAD_SIZE
    )

    # [seq_len, head_size]
    # zero out keys that aren't part of the sequence
    mask = tl.arange(0, MAX_CONTEXT_LEN)[:, None] < context_len
    kv_offs = start_tok_offset[:, None] + tl.arange(0, HEAD_SIZE)[None, :]
    keys = tl.load(scratchpad_key_ptr + kv_offs, mask=mask, other=0.0)
    values = tl.load(scratchpad_value_ptr + kv_offs, mask=mask, other=0.0)

    # keys shape  [seq_len x head_size], query shape = [head_size]
    # Can't do below b/c minimum size on all dimensions is 16
    # scores = tl.dot(query_head[None, :], keys.T)
    scores = scale * tl.sum(keys * query_head[None, :], axis=1)

    # This mask is necessary b/c even though we mask out the keys on load
    # that just results in 0s in the attention dot product,
    # which then get softmaxed and result in non-zero values
    # in the softmax output (which is wrong)
    # -inf guarantees that the softmax output will be 0 for masked values
    mask = tl.full([MAX_CONTEXT_LEN], -float("inf"), dtype=tl.float32)
    cond = tl.arange(0, MAX_CONTEXT_LEN) < context_len
    scores_masked = tl.where(cond, scores, mask)

    # do a numerically stable softmax on the scores
    scores_minus_max = scores_masked - tl.max(scores_masked, axis=0)
    numerator = tl.exp(scores_minus_max)
    denominator = tl.sum(numerator, axis=0)
    logits = numerator / denominator

    # output_offset = seq_idx * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE
    # tl.store(output_ptr + output_offset + tl.arange(0, HEAD_SIZE), logits)

    weighted_values = tl.sum(values * logits[:, None], axis=0)

    output_offset = seq_idx * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE

    tl.store(output_ptr + output_offset + tl.arange(0, HEAD_SIZE), weighted_values)


def paged_attention_triton_v1(
    output,
    query,
    key_cache,
    value_cache,
    scale,
    block_tables,
    context_lens,
    block_size,
    num_seqs,
    seq_lens,
    num_query_heads,
    max_seq_len,
    max_num_blocks_per_seq,
    head_size,
    num_kv_heads,
    scratchpad_key,
    scratchpad_value,
):

    paged_attention_v1[(num_seqs, num_query_heads)](
        scratchpad_key_ptr=scratchpad_key,
        scratchpad_value_ptr=scratchpad_value,
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_tables,
        context_lens_ptr=context_lens,
        scale=scale,
        num_seqs=num_seqs,
        num_heads=num_query_heads,
        cache_block_stride=key_cache.stride(0),
        MAX_CONTEXT_LEN=max_seq_len,
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        MAX_NUM_BLOCKS_PER_SEQ=max_num_blocks_per_seq,
    )

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables_lst[i]
        seq_len = int(seq_lens_lst[i])

        keys_lst: List[torch.Tensor] = []
        values_lst: List[torch.Tensor] = []
        for j in range(seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size
            k = key_cache[block_number, :, :, block_offset]
            k = k.reshape(num_kv_heads, head_size)
            keys_lst.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)

        # why?
        # torch.testing.assert_close(scratchpad_key[i], keys)
        # torch.testing.assert_close(scratchpad_value[i], values)
