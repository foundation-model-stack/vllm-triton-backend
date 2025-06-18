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
from third_party.vedantroy_paged_attention import paged_attention_triton_v1
from ibm_triton_lib.kernels.legacy.triton_prefix_prefill import context_attention_fwd
from .base import DecodeCaller, PrefixPrefillCaller


class BaselineTritonCaller(DecodeCaller):
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
        alibi_slopes,  # unused
        kv_cache_dtype,  # unused
    ):
        num_query_heads = query.shape[1]
        num_kv_heads = key_cache.shape[1]
        block_size = key_cache.shape[3]
        max_num_blocks_per_seq = block_tables.shape[1]
        head_size = key_cache.shape[2]

        scratchpad_key = torch.zeros(
            (num_seqs, max_seq_len, num_query_heads, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        scratchpad_value = torch.zeros_like(scratchpad_key)

        call_func_under_test = lambda: paged_attention_triton_v1(
            output,
            query,
            key_cache,
            value_cache,
            scale,
            block_tables,
            seq_lens,
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
        )

        return call_func_under_test


class BaselineTritonPrefixPrefillCaller(PrefixPrefillCaller):
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
        num_blocks = key_cache.shape[0]

        max_query_len = max(query_lens)
        k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=query.device)

        def call_and_process_output():
            return context_attention_fwd(
                q=query,
                k=key,
                v=value,
                o=output,
                kv_cache_dtype="fp16",  # TODO
                k_cache=key_cache,
                v_cache=value_cache,
                b_loc=block_tables,
                b_start_loc=start_loc,
                b_seq_len=seq_lens,
                # b_ctx_len=ctx_lens,  # FIXME: only in v0.7.3, not in main
                max_input_len=max_query_len,
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
