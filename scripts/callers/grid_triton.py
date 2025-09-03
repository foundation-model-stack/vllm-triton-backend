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
import triton

from ibm_triton_lib.kernels import unified_attention_grid
from .base import PrefixPrefillCaller


class GridTriton3dAttentionCaller(PrefixPrefillCaller):
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
    
        block_size = value.shape[1]
        num_seqs = len(seq_lens)
        num_query_heads = query.shape[1]
        num_kv_heads = key.shape[2]
        num_queries_per_kv = num_query_heads // num_kv_heads
        head_size = query.shape[2]

        query_lens = torch.diff(start_loc)
        if max_query_len == 1:
            num_decodes = len(seq_lens)
        else:
            num_decodes = torch.argmax((query_lens != 1).int()).item()
        
        BLOCK_M_PREFILL = 64
        BLOCK_M_DECODE  = 16
        BLOCK_Q_PREFILL = BLOCK_M_PREFILL * num_kv_heads // num_query_heads
        BLOCK_Q_DECODE  = BLOCK_M_DECODE  * num_kv_heads // num_query_heads

        block_q_seq_boundaries = torch.cumsum(torch.cat([torch.tensor([0], dtype=query_lens.dtype, device=query_lens.device), torch.ceil(query_lens[num_decodes:] / BLOCK_Q_PREFILL).to(torch.int)]), dim=0)
        num_q_blocks = block_q_seq_boundaries[-1].item()

        # use_split_kv = (num_q_blocks * self.num_heads_kv < 128)
        use_split_kv = force_selection == 3

        NUM_SEGMENTS=16

        if use_split_kv:
            segm_output = torch.empty(
                num_decodes,
                num_query_heads,
                NUM_SEGMENTS,
                triton.next_power_of_2(head_size),
                dtype=torch.float32,
                device=seq_lens.device,
            )
            segm_max = torch.empty(
                num_decodes,
                num_query_heads,
                NUM_SEGMENTS,
                dtype=torch.float32,
                device=seq_lens.device,
            )
            segm_expsum = torch.empty(
                num_decodes,
                num_query_heads,
                NUM_SEGMENTS,
                dtype=torch.float32,
                device=seq_lens.device,
            )
        else:
            segm_output = None
            segm_max = None
            segm_expsum = None

        if use_split_kv:
            assert num_decodes == num_seqs, "3d can only do decodes"

        def call_and_process_output():
            # k must have shape (num_blocks, page_block_size, num_heads_k, head_size)
            return unified_attention_grid(
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
                use_split_kv=use_split_kv,
                num_decodes=num_decodes,
                segm_output=segm_output,
                segm_max=segm_max,
                segm_expsum=segm_expsum,
                BLOCK_M_PREFILL=BLOCK_M_PREFILL,
                BLOCK_Q_PREFILL=BLOCK_Q_PREFILL,
                BLOCK_M_DECODE=BLOCK_M_DECODE,
                BLOCK_Q_DECODE=BLOCK_Q_DECODE,
                num_q_blocks=num_q_blocks,
                block_q_seq_boundaries=block_q_seq_boundaries
            )

        return call_and_process_output

    @staticmethod
    def requires_allocated_output() -> bool:
        return True


class GridTriton2dAttentionCaller(GridTriton3dAttentionCaller):
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
        force_selection=2,
    ):

        return GridTriton3dAttentionCaller.make_call_func(
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

