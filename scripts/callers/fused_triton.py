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
from ibm_triton_lib.kernels.legacy import fused_chunked_prefill_paged_decode_25d
from .base import PrefixPrefillCaller, DecodeCaller


class FusedTritonDecodeOnlyCaller(DecodeCaller):
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
        """
        query: shape = [num_tokens, num_heads, head_size]
        key: shape = [num_tokens, num_kv_heads, head_size]
        value: shape = [num_tokens, num_kv_heads, head_size]
        kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
        """

        query_lens = [1] * num_seqs
        b_query_lens = torch.tensor(query_lens, dtype=torch.int)
        b_start_loc = torch.cumsum(
            torch.tensor([0] + query_lens, dtype=torch.int), dim=0, dtype=torch.int
        )

        max_query_len = query_lens.max()
        # print(query.shape)
        # print(key_cache.shape)
        # print(value_cache.shape)
        k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=query.device)

        def call_and_process_output():
            return fused_chunked_prefill_paged_decode_25d(
                query=query,
                key=key_cache,  # would break, just here for benchmarking
                value=value_cache,  # would break, just here for benchmarking
                output=output,
                kv_cache_dtype=kv_cache_dtype,  # TODO
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_tables,
                query_start_loc=b_start_loc,
                seq_lens=seq_lens,
                max_query_len=max_query_len,
                k_scale=k_scale,
                v_scale=v_scale,
                alibi_slopes=None,  # TODO
                sliding_window=None,  # TODO
                sm_scale=1.0,  # would break, just here for benchmarking
            )

        return call_and_process_output

    @staticmethod
    def requires_allocated_output() -> bool:
        return True


class FusedTritonChunkedPrefixPrefill25dCaller(PrefixPrefillCaller):
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
        head_size = key_cache.shape[3]
        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]

        max_query_len = query_lens.max()
        print(start_loc)
        k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=query.device)

        def call_and_process_output():
            return fused_chunked_prefill_paged_decode_25d(
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
                sm_scale=softmax_scale,
            )

        return call_and_process_output

    @staticmethod
    def requires_allocated_output() -> bool:
        return True
