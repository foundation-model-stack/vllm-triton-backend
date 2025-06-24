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
from ibm_triton_lib.kernels.legacy import paged_attention_fp8_3d
from .base import DecodeCaller


class TritonFp8Caller(DecodeCaller):
    @staticmethod
    def make_call_func(
        output,
        query,
        key_cache,
        value_cache,
        num_seqs,
        seq_lens,
        max_seq_len,  # unused
        scale,
        block_tables,
        alibi_slopes,
        kv_cache_dtype,  # unused
    ):
        num_query_heads = query.shape[1]
        num_kv_heads = key_cache.shape[1]
        block_size = key_cache.shape[3]
        num_queries_per_kv = num_query_heads // num_kv_heads
        head_size = key_cache.shape[2]

        key_cache_ykt = key_cache.permute(0, 1, 3, 2).contiguous()
        value_cache_ykt = value_cache.permute(0, 1, 3, 2).contiguous()

        call_func_under_test = lambda: paged_attention_fp8_3d(
            output,
            query,
            key_cache_ykt,
            value_cache_ykt,
            scale,
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
