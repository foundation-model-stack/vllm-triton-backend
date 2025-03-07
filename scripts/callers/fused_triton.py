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
from ibm_triton_lib.kernels import fused_chunked_prefill_paged_decode_25d
from .base import PrefixPrefillCaller


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
        kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
        """

        max_query_len = max(query_lens)
        # print(query.shape)
        # print(key_cache.shape)
        # print(value_cache.shape)
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
