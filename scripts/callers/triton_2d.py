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
from ibm_triton_lib.kernels.legacy import (
    paged_attention_2d,
    chunked_prefill_paged_decode,
)
from .base import DecodeCaller, PrefixPrefillCaller


class Triton2dAttentionDecodeCaller(DecodeCaller):
    @staticmethod
    def make_call_func(
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
    ):
        num_query_heads = query.shape[1]
        num_kv_heads = key_cache.shape[1]
        block_size = key_cache.shape[3]
        num_queries_per_kv = num_query_heads // num_kv_heads
        max_num_blocks_per_seq = block_tables.shape[1]
        head_size = key_cache.shape[2]

        # Using default kv_scale
        k_scale = v_scale = torch.ones(1, device=query.device)

        call_func_under_test = lambda: paged_attention_2d(
            output,
            query,
            key_cache,
            value_cache,
            scale,
            k_scale,
            v_scale,
            kv_cache_dtype,
            block_tables,
            seq_lens,
            alibi_slopes,
            block_size,
            num_seqs,
            num_query_heads,
            num_queries_per_kv,
            head_size,
        )

        return call_func_under_test


class Triton2dChunkedPrefillCaller(PrefixPrefillCaller):
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

        needs to be converted to
        K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
        V_cache[num_blocks, num_kv_heads, head_size, block_size]

        Returns:
            shape = [num_tokens, num_heads, head_size]
        """

        max_query_len = max(query_lens)
        k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=query.device)

        def call_and_process_output():
            return chunked_prefill_paged_decode(
                query=query,
                key=key,
                value=value,
                output=output,
                kv_cache_dtype="fp16",  # TODO
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_tables,
                query_start_loc=start_loc,
                seq_lens=seq_lens,
                max_query_len=max_query_len,
                k_scale=k_scale,
                v_scale=v_scale,
                alibi_slopes=None,  # TODO
                sliding_window=None,  # TODO
                scale=softmax_scale,
            )

        return call_and_process_output

    @staticmethod
    def requires_allocated_output() -> bool:
        return True
