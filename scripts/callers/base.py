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


class DecodeCaller:
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
        raise NotImplementedError

    @classmethod
    def select_output(cls, x, y):
        if cls.requires_allocated_output():
            # default behaviour is in-place
            return x
        else:
            return y

    @staticmethod
    def requires_allocated_output() -> bool:
        # default behaviour is in-place -> so yes
        return True


class PrefillCaller:
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
        raise NotImplementedError

    @classmethod
    def select_output(cls, x, y):
        if cls.requires_allocated_output():
            # default behaviour is in-place
            return x
        else:
            return y

    @staticmethod
    def requires_allocated_output() -> bool:
        # default behaviour is in-place -> so yes
        return True
