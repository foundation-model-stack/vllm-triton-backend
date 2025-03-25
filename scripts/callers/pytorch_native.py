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


import torch
from .base import PrefillCaller


# based on https://github.com/pytorch/pytorch/blob/6055a4f612782ca944f2e0465f7497b7f18de4e9/torch/nn/functional.py#L5732
def scaled_dot_product_attention(
    query,
    key,
    value,
    scale_factor,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    enable_gqa=False,
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)  # .to(query.dtype)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class PytorchNativeAttentionPrefillCaller(PrefillCaller):
    @staticmethod
    def make_call_func(
        output,
        query,
        key_cache,
        value_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        # kv_cache_dtype,  # unused
    ):
        # with varlen
        # q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        # k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        # v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        # cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
        #    of the sequences in the batch, used to index into q.
        # cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
        #    of the sequences in the batch, used to index into kv.
        # max_seqlen_q: int. Maximum query sequence length in the batch.
        # max_seqlen_k: int. Maximum key sequence length in the batch.
        # out: (total, nheads, headdim).

        print(query.shape)
        print(key_cache.shape)

        num_query_heads = query.shape[1]
        num_kv_heads = key_cache.shape[1]
        head_size = key_cache.shape[2]
        dtype = value_cache.dtype
        tdevice = value_cache.device

        num_queries_per_kv = num_query_heads // num_kv_heads
        enable_gqa = False
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            enable_gqa = True
            # key = torch.repeat_interleave(key, num_queries_per_kv, dim=1)
            # value = torch.repeat_interleave(value, num_queries_per_kv, dim=1)

        num_seqs = len(cu_seqlens_q) - 1
        q_len = max_seqlen_q
        max_seq_len = max_seqlen_k

        # print(max_seqlen_q)
        # print(max_seqlen_k)

        query_torch = torch.empty(
            num_seqs, num_query_heads, q_len, head_size, dtype=dtype, device=tdevice
        )
        key_torch = torch.empty(
            num_seqs, num_kv_heads, max_seq_len, head_size, dtype=dtype, device=tdevice
        )
        value_torch = torch.empty(
            num_seqs, num_kv_heads, max_seq_len, head_size, dtype=dtype, device=tdevice
        )

        # print(query_torch.shape)
        # print(key_torch.shape)

        for i in range(num_seqs):
            start_idx = cu_seqlens_q[i]
            end_idx = cu_seqlens_q[i + 1]
            seq_len = end_idx - start_idx
            # print(f"{start_idx} to {end_idx} ({seq_len} tokens)")

            query_torch[i].copy_(query[start_idx:end_idx].transpose(0, 1))
            key_torch[i].copy_(key_cache[start_idx:end_idx].transpose(0, 1))
            value_torch[i].copy_(value_cache[start_idx:end_idx].transpose(0, 1))
            # TODO: fill with 0?
            # no, compare varlen with it would be unfair, IMHO

        def call_and_process_output():
            return scaled_dot_product_attention(
                query=query_torch,
                key=key_torch,
                value=value_torch,
                is_causal=causal,
                enable_gqa=enable_gqa,
                scale_factor=softmax_scale,
            )

        return call_and_process_output

    @classmethod
    def select_output(cls, x, y):
        # in: (num_seqs, num_query_heads, q_len, head_size)
        # out: (total, nheads, headdim)
        print(y.shape)
        num_seqs, num_query_heads, q_len, head_size = y.shape
        out = y.transpose(1, 2).reshape(-1, num_query_heads, head_size)
        print(out.shape)
        # return y.squeeze(1)
        return out

    @staticmethod
    def requires_allocated_output() -> bool:
        return False
