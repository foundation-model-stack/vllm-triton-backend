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
from .base import DecodeCaller


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
