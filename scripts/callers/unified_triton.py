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

from ibm_triton_lib.kernels import unified_attention
from .base import PrefixPrefillCaller


class UnifiedTriton3dAttentionCaller(PrefixPrefillCaller):
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
        force_selection=3,
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

        avg_seqlen_q = query_lens.to(torch.float).mean()
        avg_seqlen_k = seq_lens.to(torch.float).mean()

        def call_and_process_output():
            # k must have shape (num_blocks, page_block_size, num_heads_k, head_size)
            return unified_attention(
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
                window_size=(-1, -1),
                block_table=block_tables,
                softcap=0,
                q_descale=None,
                k_descale=None,  # TODO?
                v_descale=None,  # TODO?
                alibi_slopes=None,
                avg_seqlen_q=avg_seqlen_q,
                avg_seqlen_k=avg_seqlen_k,
                force_selection=force_selection,
            )

        return call_and_process_output

    @staticmethod
    def requires_allocated_output() -> bool:
        return True


class UnifiedTriton2dAttentionCaller(UnifiedTriton3dAttentionCaller):
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
        force_selection=3,
    ):

        return UnifiedTriton3dAttentionCaller.make_call_func(
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
            force_selection=2,
        )


class UnifiedTritonAutoAttentionCaller(UnifiedTriton3dAttentionCaller):
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
        force_selection=3,
    ):

        return UnifiedTriton3dAttentionCaller.make_call_func(
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
            force_selection=None,
        )  # none triggers vllm default behaviour
