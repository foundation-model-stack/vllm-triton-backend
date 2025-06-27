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


from typing import List, Optional, Tuple, Union
from vllm.platforms import current_platform
import torch
from vllm.utils import get_kv_cache_torch_dtype


def _generate_random_fp8(
    tensor: torch.Tensor,
    low: float,
    high: float,
) -> None:
    # NOTE(zhaoyang): Due to NaN and Inf representation for fp8 data type,
    # it may occur Inf or NaN if we directly use torch.randint
    # to generate random data for fp8 data.
    # For example, s.11111.00 in fp8e5m2 format represents Inf.
    #     | E4M3        | E5M2
    # -----|-------------|-------------------
    # Inf | N/A         | s.11111.00
    # NaN | s.1111.111  | s.11111.{01,10,11}
    from vllm import _custom_ops as ops

    tensor_tmp = torch.empty_like(tensor, dtype=torch.float16)
    tensor_tmp.uniform_(low, high)
    ops.convert_fp8(tensor, tensor_tmp)
    del tensor_tmp


def create_kv_caches_with_random(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    max_value: float,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
    alignment_optimization=False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    current_platform.seed_everything(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    key_cache_shape = (num_blocks, num_heads, head_size, block_size)
    if alignment_optimization:
        x = 16 // torch.tensor([], dtype=torch_dtype).element_size()
        key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(-max_value, max_value)
        elif cache_dtype == "fp8":
            # FIXME
            _generate_random_fp8(key_cache, -max_value, max_value)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(
            size=value_cache_shape, dtype=torch_dtype, device=device
        )
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(-max_value, max_value)
        elif cache_dtype == "fp8":
            # FIXME
            _generate_random_fp8(value_cache, -max_value, max_value)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def ref_single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0]

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
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(seq_len).int()
            alibi_bias = (position_ids - seq_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)


def ref_multi_query_kv_attention(
    cu_seq_lens: List[int],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype: torch.dtype,
    num_kv_heads: float,
    num_query_heads: float,
) -> torch.Tensor:

    num_queries_per_kv = num_query_heads // num_kv_heads
    if num_queries_per_kv > 1:
        # Handle MQA and GQA
        key = torch.repeat_interleave(key, num_queries_per_kv, dim=1)
        value = torch.repeat_interleave(value, num_queries_per_kv, dim=1)

    num_seqs = len(cu_seq_lens) - 1
    ref_outputs: List[torch.Tensor] = []
    for i in range(num_seqs):
        start_idx = cu_seq_lens[i]
        end_idx = cu_seq_lens[i + 1]
        seq_len = end_idx - start_idx

        # Create attention mask.
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype), diagonal=1)
        attn_mask = attn_mask * torch.finfo(dtype).min
        attn_mask = attn_mask.to(dtype=dtype)

        ref_output = ref_masked_attention(
            query[start_idx:end_idx],
            key[start_idx:end_idx],
            value[start_idx:end_idx],
            scale,
            attn_mask=attn_mask,
        )
        ref_outputs.append(ref_output)

    return torch.cat(ref_outputs, dim=0)


def ref_prefix_prefill(
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    ctx_lens: torch.Tensor,
    query_lens: torch.Tensor,
    start_loc: torch.Tensor,
    batch_size,
    scale: float,
    dtype: torch.dtype,
):
    """
    query: shape = [num_tokens, num_heads, head_size]
    key: shape = [num_tokens, num_kv_heads, head_size]
    value: shape = [num_tokens, num_kv_heads, head_size]
    k_cache = [num_blocks, block_size, num_kv_heads, head_size]
    v_cache = [num_blocks, block_size, num_kv_heads, head_size]
    Returns:
        shape = [num_tokens, num_heads, head_size]
    """
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[2]
    head_size = value_cache.shape[3]
    block_size = value_cache.shape[1]
    num_seqs = batch_size

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()
    ctx_lens_lst = ctx_lens.cpu().tolist()
    query_lens_lst = query_lens.cpu().tolist()
    start_loc_lst = start_loc.cpu().tolist()
    ref_outputs: List[torch.Tensor] = []

    for i in range(num_seqs):
        cur_batch_seq_len = seq_lens_lst[i]
        cur_batch_in_all_start_index = start_loc_lst[i]
        cur_batch_in_all_stop_index = start_loc_lst[i + 1]
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        assert cur_batch_query_len == query_lens_lst[i]
        cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

        if cur_batch_query_len == 1:
            # normal decode
            q = query[cur_batch_in_all_start_index:cur_batch_in_all_stop_index]
            block_table = block_tables_lst[i]
            seq_len = int(ctx_lens_lst[i])
            keys_lst: List[torch.Tensor] = []
            values_lst: List[torch.Tensor] = []
            for j in range(seq_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size
                k = key_cache[block_number, block_offset, :, :]
                keys_lst.append(k)
                v = value_cache[block_number, block_offset, :, :]
                values_lst.append(v)
            reconstr_keys = torch.stack(keys_lst, dim=0)
            reconstr_values = torch.stack(values_lst, dim=0)
            if num_queries_per_kv > 1:
                # Handle MQA and GQA
                reconstr_keys = torch.repeat_interleave(
                    reconstr_keys, num_queries_per_kv, dim=1
                )
                reconstr_values = torch.repeat_interleave(
                    reconstr_values, num_queries_per_kv, dim=1
                )
            out = ref_masked_attention(q, reconstr_keys, reconstr_values, scale)
            ref_outputs.append(out)
        elif cur_batch_ctx_len == 0:
            # normal prefill
            seq_len = int(query_lens_lst[i])
            # Create attention mask.
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=dtype), diagonal=1
            )
            attn_mask = attn_mask * torch.finfo(dtype).min
            attn_mask = attn_mask.to(dtype=dtype)

            key_to_use = key[cur_batch_in_all_start_index:cur_batch_in_all_stop_index]
            value_to_use = value[
                cur_batch_in_all_start_index:cur_batch_in_all_stop_index
            ]
            if num_queries_per_kv > 1:
                # Handle MQA and GQA
                key_to_use = torch.repeat_interleave(
                    key_to_use, num_queries_per_kv, dim=1
                )
                value_to_use = torch.repeat_interleave(
                    value_to_use, num_queries_per_kv, dim=1
                )

            out = ref_masked_attention(
                query[cur_batch_in_all_start_index:cur_batch_in_all_stop_index],
                key_to_use,
                value_to_use,
                scale,
                attn_mask=attn_mask,
            )
            ref_outputs.append(out)
        else:
            # prefix prefill
            # construct continuous context
            block_table = block_tables_lst[i]
            seq_len = int(seq_lens_lst[i])
            ctx_len = int(ctx_lens_lst[i])
            keys_lst: List[torch.Tensor] = []
            values_lst: List[torch.Tensor] = []
            for j in range(ctx_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size
                k = key_cache[block_number, block_offset, :, :]
                keys_lst.append(k)
                v = value_cache[block_number, block_offset, :, :]
                values_lst.append(v)
            reconstructed_keys = torch.stack(keys_lst, dim=0)
            reconstructed_values = torch.stack(values_lst, dim=0)
            linear_key = key[cur_batch_in_all_start_index:cur_batch_in_all_stop_index]
            linear_value = value[
                cur_batch_in_all_start_index:cur_batch_in_all_stop_index
            ]
            if num_queries_per_kv > 1:
                # Handle MQA and GQA
                reconstructed_keys = torch.repeat_interleave(
                    reconstructed_keys, num_queries_per_kv, dim=1
                )
                reconstructed_values = torch.repeat_interleave(
                    reconstructed_values, num_queries_per_kv, dim=1
                )
                linear_key = torch.repeat_interleave(
                    key[cur_batch_in_all_start_index:cur_batch_in_all_stop_index],
                    num_queries_per_kv,
                    dim=1,
                )
                linear_value = torch.repeat_interleave(
                    value[cur_batch_in_all_start_index:cur_batch_in_all_stop_index],
                    num_queries_per_kv,
                    dim=1,
                )
            all_keys = [
                reconstructed_keys,
                linear_key,
            ]
            all_values = [reconstructed_values, linear_value]
            all_keys_t = torch.cat(all_keys, dim=0)
            all_values_t = torch.cat(all_values, dim=0)
            # Create attention mask.
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=dtype), diagonal=1
            )
            attn_mask = attn_mask * torch.finfo(dtype).min
            attn_mask = attn_mask.to(dtype=dtype)
            # compute attention
            out = ref_masked_attention(
                query[cur_batch_in_all_start_index:cur_batch_in_all_stop_index],
                all_keys_t,
                all_values_t,
                scale,
            )
            ref_outputs.append(out)
    return torch.cat(ref_outputs, dim=0)


def ref_reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    num_tokens: int,
):
    """
    key: shape = [num_tokens, num_kv_heads, head_size]
    value: shape = [num_tokens, num_kv_heads, head_size]
    key_cache = [num_blocks, block_size, num_kv_heads, head_size]
    value_cache = [num_blocks, block_size, num_kv_heads, head_size]
    """

    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies_lst = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets_lst = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies_lst[i]
        block_offset = block_offsets_lst[i]
        key_cache[block_idx, block_offset, :, :] = key[i]
        value_cache[block_idx, block_offset, :, :] = value[i]


def ref_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    num_tokens: int,
):
    """
    key: shape = [num_tokens, num_kv_heads, head_size]
    value: shape = [num_tokens, num_kv_heads, head_size]
    key_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    value_cache[num_blocks, num_kv_heads, head_size, block_size]
    """

    reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies_lst = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets_lst = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies_lst[i]
        block_offset = block_offsets_lst[i]
        key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        value_cache[block_idx, :, :, block_offset] = value[i]


# from https://github.com/vllm-project/vllm/blob/main/tests/kernels/attention/test_triton_unified_attention.py
def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)
