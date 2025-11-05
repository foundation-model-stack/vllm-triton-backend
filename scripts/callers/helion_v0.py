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

import os
import torch

from ibm_triton_lib.kernels import helion_attention
from .base import PrefixPrefillCaller


class HelionV0AttentionCaller(PrefixPrefillCaller):
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

        avg_seqlen_q = query_lens.to(torch.float).mean()
        avg_seqlen_k = seq_lens.to(torch.float).mean()

        block_size = value.shape[1]
        num_seqs = len(seq_lens)
        num_query_heads = query.shape[1]
        num_kv_heads = key.shape[2]
        num_queries_per_kv = num_query_heads // num_kv_heads
        head_size = query.shape[2]

        query_lens = torch.diff(start_loc)

        def call_and_process_output():
            return helion_attention(
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
                is_decode_only=bool(max_query_len == 1),
            )
        
        if os.environ.get("USE_UPSTREAM_IF_PRESENT", "0") == "1":
            try:
                from vllm.attention.ops.helion_unified_attention import helion_unified_attention as vllm_helion_attention
                print("using vllm version of helion attention")
                def call_and_process_output():
                    return vllm_helion_attention(
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
                    )
        
            except ModuleNotFoundError:
                print("cannot overwrite helion_attention: vllm not present")

        return call_and_process_output

    @staticmethod
    def requires_allocated_output() -> bool:
        return True
