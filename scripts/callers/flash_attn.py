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

if torch.version.hip:
    from flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func
else:
    from vllm.vllm_flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func
from .base import DecodeCaller, PrefillCaller, PrefixPrefillCaller


class FlashAttnDecodeCaller(DecodeCaller):
    @staticmethod
    def make_call_func(
        output,
        query,
        key_cache,
        value_cache,
        num_seqs,  # unused
        seq_lens,
        max_seq_len,  # unused
        scale,
        block_tables,
        alibi_slopes,
        kv_cache_dtype,  # unused
    ):
        def transform_kv_cache(x):
            out = torch.transpose(x, 1, 3)
            out = torch.transpose(out, 2, 3)
            return out.contiguous()

        key_cache_flash_attn = transform_kv_cache(key_cache)
        value_cache_flash_attn = transform_kv_cache(value_cache)

        q = query.unsqueeze(1)

        if torch.version.hip:
            call_func_under_test = lambda: flash_attn_with_kvcache(
                q=q,
                k_cache=key_cache_flash_attn,
                v_cache=value_cache_flash_attn,
                softmax_scale=scale,
                causal=True,
                cache_seqlens=seq_lens,
                window_size=(-1, 1),
                block_table=block_tables,
                softcap=0,
                alibi_slopes=alibi_slopes,
            )
        else:
            call_func_under_test = lambda: flash_attn_with_kvcache(
                q=q,
                k_cache=key_cache_flash_attn,
                v_cache=value_cache_flash_attn,
                out=None,
                softmax_scale=scale,
                causal=True,
                cache_seqlens=seq_lens,
                window_size=(-1, 1),
                block_table=block_tables,
                softcap=0,
                alibi_slopes=alibi_slopes,
            )

        return call_func_under_test

    @classmethod
    def select_output(cls, x, y):
        return y.squeeze(1)

    @staticmethod
    def requires_allocated_output() -> bool:
        return False


class FlashAttnPrefillCaller(PrefillCaller):
    @staticmethod
    def make_call_func(
        output,  # unused
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

        def call_and_process_output():
            return flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return call_and_process_output

    @staticmethod
    def requires_allocated_output() -> bool:
        return False


class FlashAttnPrefixPrefillCaller(PrefixPrefillCaller):
    @staticmethod
    def make_call_func(
        output,
        query,
        key_cache,
        value_cache,
        key,
        value,
        block_tables,
        seq_lens,
        ctx_lens,
        query_lens,
        start_loc,
        seq_start_loc,
        softmax_scale,
        # kv_cache_dtype,  # unused
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

        max_query_len = query_lens.max()
        max_seqlen = seq_lens.max()

        def call_and_process_output():
            # k must have shape (num_blocks, page_block_size, num_heads_k, head_size)
            return flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                out=output,
                cu_seqlens_q=start_loc,
                max_seqlen_q=max_query_len,
                seqused_k=seq_lens,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                causal=True,
                block_table=block_tables,
                # window_size=(-1, 1),
                # softcap=0,
                # fa_version=2, # TODO
            )

        return call_and_process_output

    @staticmethod
    def requires_allocated_output() -> bool:
        return True
