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

"""
Triton-only attention backend.
===============================

This backend uses flash attention (v2) and paged attention all in Triton.

Based on the ROCm backend (https://github.com/vllm-project/vllm/blob/v0.7.2/vllm/attention/backends/rocm_flash_attn.py)
and the PagedAttention implementation (https://github.com/vllm-project/vllm/blob/v0.7.2/vllm/attention/ops/paged_attn.py)

"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch
import os

import vllm.envs as envs

# TODO: currently needed for advance_step and reshape_and_cache
from vllm import _custom_ops as ops
from vllm.triton_utils import HAS_TRITON

# TODO: better idea?
assert HAS_TRITON

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionType,
    AttentionLayer,
)
from vllm.attention.backends.utils import CommonAttentionState, CommonMetadataBuilder
from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


# TODO: use triton reshape and cache
# from vllm.attention.ops.reshape_and_cache import (
#     reshape_and_cache as triton_reshape_and_cache,
# )
from ibm_triton_lib.kernels import (
    paged_attention_2d,
    paged_attention_3d,
    prefill_flash_attention,
)
from vllm.attention.ops.prefix_prefill import context_attention_fwd


class TritonAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "zrl-triton-attn"

    @staticmethod
    def get_impl_cls() -> Type["TritonAttentionImpl"]:
        return TritonAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return TritonAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size * num_kv_heads * head_size)


@dataclass
class TritonAttentionMetadata(AttentionMetadata):
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor] = None

    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int] = None

    _cached_prefill_metadata: Optional["TritonAttentionMetadata"] = None
    _cached_decode_metadata: Optional["TritonAttentionMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None

    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None

    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    @property
    def prefill_metadata(self) -> Optional["TritonAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None
        assert self.block_tables is not None

        self._cached_prefill_metadata = TritonAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=self.slot_mapping[: self.num_prefill_tokens],
            multi_modal_placeholder_index_maps=self.multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation=self.enable_kv_scales_calculation,
            seq_lens=self.seq_lens[: self.num_prefills],
            seq_lens_tensor=self.seq_lens_tensor[: self.num_prefills],
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=(
                None
                if self.query_start_loc is None
                else self.query_start_loc[: self.num_prefills + 1]
            ),
            seq_start_loc=(
                None
                if self.seq_start_loc is None
                else self.seq_start_loc[: self.num_prefills + 1]
            ),
            context_lens_tensor=(
                None
                if self.context_lens_tensor is None
                else self.context_lens_tensor[: self.num_prefills]
            ),
            block_tables=self.block_tables[: self.num_prefills],
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["TritonAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None

        self._cached_decode_metadata = TritonAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens :],
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            seq_lens=None,
            seq_lens_tensor=self.seq_lens_tensor[self.num_prefills :],
            max_query_len=None,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            query_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=None,
            block_tables=self.block_tables[self.num_prefills :],
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
        )
        # Batch may be composed of prefill|decodes, adjust query start indices
        # to refer to the start of decodes when the two are split apart.
        # E.g. in tokens:[3 prefills|6 decodes], query_start_loc=[3,9] => [0,6].
        if self._cached_decode_metadata.query_start_loc is not None:
            qs = self._cached_decode_metadata.query_start_loc
            self._cached_decode_metadata.query_start_loc = qs - qs[0]
        return self._cached_decode_metadata

    def advance_step(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        sampled_token_ids: Optional[torch.Tensor],
        block_size: int,
        num_seqs: int,
        num_queries: int,
        turn_prefills_into_decodes: bool = False,
    ):
        """
        Update metadata in-place to advance one decode step.
        """

        assert not turn_prefills_into_decodes, (
            "Multi Step Chunked prefill is not supported with triton_attn yet."
            "turn_prefills_into_decodes is a Multi-Step + Chunked-Prefill "
            "specific parameter."
        )

        # When using cudagraph, the num_seqs is padded to the next captured
        # batch sized, but num_queries tracks the actual number of requests in
        # the batch. For --enforce-eager mode, num_seqs == num_queries
        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert self.use_cuda_graph

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs,)

        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs,)
        assert self.max_query_len == 1
        assert self.max_prefill_seq_len == 0
        assert self.max_decode_seq_len == max(self.seq_lens)

        assert self.query_start_loc is not None
        assert self.query_start_loc.shape == (num_queries + 1,)
        assert self.seq_start_loc is not None
        assert self.seq_start_loc.shape == (num_seqs + 1,)

        assert self.context_lens_tensor is not None
        assert self.context_lens_tensor.shape == (num_queries,)

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)

        # TODO: add triton implementation
        ops.advance_step_flashattn(
            num_seqs=num_seqs,
            num_queries=num_queries,
            block_size=block_size,
            input_tokens=model_input.input_tokens,
            sampled_token_ids=sampled_token_ids,
            input_positions=model_input.input_positions,
            seq_lens=self.seq_lens_tensor,
            slot_mapping=self.slot_mapping,
            block_tables=self.block_tables,
        )


class TritonAttentionMetadataBuilder(CommonMetadataBuilder[TritonAttentionMetadata]):

    _metadata_cls = TritonAttentionMetadata


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    dtype: torch.dtype,
    seq_lens: Optional[List[int]],
    make_attn_mask: bool = True,
) -> List[torch.Tensor]:
    attn_biases = []
    if seq_lens:
        for seq_len in seq_lens:
            bias = torch.arange(seq_len, dtype=dtype)
            # NOTE(zhuohan): HF uses
            #     `bias = bias[None, :].repeat(seq_len, 1)`
            # here. We find that both biases give the same results, but
            # the bias below more accurately follows the original ALiBi
            # paper.
            bias = bias[None, :] - bias[:, None]

            num_heads = alibi_slopes.shape[0]
            bias = bias[None, :].repeat((num_heads, 1, 1)).to(alibi_slopes.device)
            bias.mul_(alibi_slopes[:, None, None])
            if make_attn_mask:
                inf_mask = (
                    torch.empty((1, seq_len, seq_len), dtype=bias.dtype)
                    .fill_(-torch.inf)
                    .triu_(diagonal=1)
                    .to(alibi_slopes.device)
                )
                attn_biases.append((bias + inf_mask).to(dtype))
            else:
                attn_biases.append(bias.to(dtype))

    return attn_biases


def _get_seq_len_block_table_args(
    attn_metadata: TritonAttentionMetadata,
    attn_type: str,
) -> tuple:
    """
    The particular choice of sequence-length
    attributes which should be extracted from attn_metadata is dependent
    on the type of attention operation.

    Decoder attn -> select entirely decoder self-attention-related fields
    Encoder/decoder cross-attn -> select encoder sequence lengths
    Encoder attn -> select encoder sequence lengths fields

    Arguments:

    * attn_metadata: Attention metadata structure associated with attention op
    * attn_type: encoder attention, decoder self-attention,
                encoder/decoder cross-attention

    Returns:

    * Appropriate sequence-lengths tensors for query and key
    * Appropriate max sequence-length scalar
    """

    partial_prefix_sum = 0
    if attn_type == AttentionType.ENCODER:
        assert attn_metadata.encoder_seq_lens is not None
        assert attn_metadata.encoder_seq_lens_tensor is not None
        query_seq_start_loc = torch.tensor(
            [0]
            + [
                partial_prefix_sum := partial_prefix_sum + i
                for i in attn_metadata.encoder_seq_lens
            ],
            device=attn_metadata.encoder_seq_lens_tensor.device,
            dtype=attn_metadata.encoder_seq_lens_tensor.dtype,
        )
        causal_mask = False

        # No block tables associated with encoder attention
        return (
            query_seq_start_loc,
            attn_metadata.max_encoder_seq_len,
            query_seq_start_loc,
            attn_metadata.max_encoder_seq_len,
            attn_metadata.encoder_seq_lens,
            causal_mask,
        )
    elif attn_type == AttentionType.DECODER:
        # Decoder self-attention
        # Choose max_seq_len based on whether we are in prompt_run
        assert attn_metadata.seq_lens is not None
        assert attn_metadata.seq_lens_tensor is not None
        query_seq_start_loc = torch.tensor(
            [0]
            + [
                partial_prefix_sum := partial_prefix_sum + i
                for i in attn_metadata.seq_lens
            ],
            device=attn_metadata.seq_lens_tensor.device,
            dtype=attn_metadata.seq_lens_tensor.dtype,
        )
        max_seq_len = attn_metadata.max_prefill_seq_len
        causal_mask = True

        return (
            query_seq_start_loc,
            max_seq_len,
            query_seq_start_loc,
            max_seq_len,
            attn_metadata.seq_lens,
            causal_mask,
        )
    elif attn_type == AttentionType.ENCODER_DECODER:
        assert attn_metadata.seq_lens is not None
        assert attn_metadata.encoder_seq_lens_tensor is not None
        query_start_loc = torch.tensor(
            [0]
            + [
                partial_prefix_sum := partial_prefix_sum + i
                for i in attn_metadata.seq_lens
            ],
            device=attn_metadata.encoder_seq_lens_tensor.device,
            dtype=attn_metadata.encoder_seq_lens_tensor.dtype,
        )

        partial_prefix_sum = 0
        assert attn_metadata.encoder_seq_lens is not None
        assert attn_metadata.seq_lens_tensor is not None
        key_seq_start_loc = torch.tensor(
            [0]
            + [
                partial_prefix_sum := partial_prefix_sum + i
                for i in attn_metadata.encoder_seq_lens
            ],
            device=attn_metadata.seq_lens_tensor.device,
            dtype=attn_metadata.seq_lens_tensor.dtype,
        )
        causal_mask = False

        # Enc/dec cross-attention KVs match encoder sequence length;
        # cross-attention utilizes special "cross" block tables
        return (
            query_start_loc,
            attn_metadata.max_prefill_seq_len,
            key_seq_start_loc,
            attn_metadata.max_encoder_seq_len,
            attn_metadata.seq_lens,
            causal_mask,
        )
    else:
        raise AttributeError(f"Invalid attention type {str(attn_type)}")


class TritonAttentionImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prompt_tokens -------------->|
    |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->|

    Otherwise, the layout is as follows:
    |<------------------ num_generation_tokens (M) ----------------->|
    |<--generation_0-->|..........|<--generation_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens ----------->|
    |<-prompt_0->|...|<-prompt_N-1->|<-generation_0->|...|<-generation_M-1->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError("TritonAttention does not support blocksparse attention.")

        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            self.logits_soft_cap = 0.0
        else:
            self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = (
            (sliding_window, sliding_window) if sliding_window is not None else (-1, -1)
        )
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        supported_head_sizes = self.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}."
            )

        # self.attn_func = triton_wrapper_forward
        logger.debug("Using Triton Prefill attention")
        # TODO
        if self.sliding_window != (-1, -1):
            logger.warning(
                "Triton FA does not currently support " "sliding window attention. "
            )
        logger.debug("Using Triton Paged attention")

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
        tokens, n_kv_heads, head_dim = x.shape
        return (
            x[:, :, None, :]
            .expand(tokens, n_kv_heads, n_rep, head_dim)
            .reshape(tokens, n_kv_heads * n_rep, head_dim)
        )

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 80, 96, 112, 120, 128, 192, 256]

    def split_kv_cache(
        self,
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size, -1, 1)
        # TODO: maybe enable cache alginment as in cuda?
        # x = 16 // kv_cache.element_size()
        # key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x, -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

    def write_to_paged_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
    ) -> None:
        # TODO use integrate triton reshape and cache
        #     triton_reshape_and_cache(
        #         key, value, key_cache, value_cache, slot_mapping.flatten()
        #     )
        ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        # used by cache_engine
        # TODO: add triton implementation and make non-static method
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        # used by cache_engine
        # TODO: add triton implementation and make non-static method
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention and PagedAttention.

        For decoder-only models: query, key and value must be non-None.

        For encoder/decoder models:
        * TritonAttentionImpl.forward() may be invoked for both self- and
            cross-attention layers.
        * For self-attention: query, key and value must be non-None.
        * For cross-attention:
            * Query must be non-None
            * During prefill, key and value must be non-None; key and value
              get cached for use during decode.
            * During decode, key and value may be None, since:
              (1) key and value tensors were cached during prefill, and
              (2) cross-attention key and value tensors do not grow during
                  decode

        A note on how the attn_type (attention type enum) argument impacts
        attention forward() behavior:

            * DECODER: normal decoder-only behavior;
                use decoder self-attention block table
            * ENCODER: no KV caching; pass encoder sequence
                attributes (encoder_seq_lens/encoder_seq_lens_tensor/
                max_encoder_seq_len) to kernel, in lieu of decoder
                sequence attributes (seq_lens/seq_lens_tensor/max_seq_len)
            * ENCODER_DECODER: cross-attention behavior;
                use cross-attention block table for caching KVs derived
                from encoder hidden states; since KV sequence lengths
                will match encoder sequence lengths, pass encoder sequence
                attributes to kernel (encoder_seq_lens/encoder_seq_lens_tensor/
                max_encoder_seq_len)

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
            attn_type: Select attention type, between encoder attention,
                       decoder self-attention, or encoder/decoder cross-
                       attention. Defaults to decoder self-attention,
                       which is the vLLM default generally
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        if self.attn_type != AttentionType.ENCODER and kv_cache.numel() > 0:
            key_cache, value_cache = self.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size
            )

            if key is not None and value is not None:
                # Reshape the input keys and values and store them in the
                # cache. If kv_cache is not provided, the new key and value
                # tensors are not cached. This happens during the initial
                # memory profiling run.
                self.write_to_paged_cache(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    (
                        attn_metadata.slot_mapping
                        if self.attn_type != AttentionType.ENCODER_DECODER
                        else attn_metadata.cross_slot_mapping
                    ),
                    self.kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )

        if self.attn_type != AttentionType.ENCODER:
            num_prefill_tokens = attn_metadata.num_prefill_tokens
        else:
            assert attn_metadata.num_encoder_tokens is not None
            num_prefill_tokens = attn_metadata.num_encoder_tokens

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_tokens]

        if (
            key is not None
            and value is not None
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_prefill_tokens]
            value = value[:num_prefill_tokens]

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            # normal attention and DECODER
            if self.attn_type == AttentionType.DECODER and (
                kv_cache.numel() == 0
                or prefill_meta.block_tables is None
                or prefill_meta.block_tables.numel() == 0
            ):
                (
                    query_seq_start_loc,
                    query_max_seq_len,
                    key_seq_start_loc,
                    key_max_seq_len,
                    seq_lens,
                    causal_mask,
                ) = (
                    prefill_meta.seq_start_loc,
                    prefill_meta.max_prefill_seq_len,
                    prefill_meta.seq_start_loc,
                    prefill_meta.max_prefill_seq_len,
                    attn_metadata.seq_lens,
                    True,
                )
            # prefix-enabled attention and ENCODER/ENCODER_DECODER
            else:
                (
                    query_seq_start_loc,
                    query_max_seq_len,
                    key_seq_start_loc,
                    key_max_seq_len,
                    seq_lens,
                    causal_mask,
                ) = _get_seq_len_block_table_args(prefill_meta, self.attn_type)
            # Prompt run.
            if kv_cache.numel() == 0 or prefill_meta.block_tables.numel() == 0:
                # triton attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                attn_masks = None
                if self.alibi_slopes is not None:
                    # FIXME
                    attn_masks = _make_alibi_bias(
                        self.alibi_slopes,
                        query.dtype,
                        attn_metadata.seq_lens,
                        make_attn_mask=False,
                    )  # type: ignore

                out = prefill_flash_attention(
                    query,
                    key,
                    value,
                    sm_scale=self.scale,
                    causal=True,
                    max_seqlen_q=prefill_meta.max_prefill_seq_len,
                    max_seqlen_k=prefill_meta.max_prefill_seq_len,
                    cu_seqlens_q=prefill_meta.seq_start_loc,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                )

                assert output[:num_prefill_tokens].shape == out.shape
                if output.shape[0] > num_prefill_tokens:
                    output[:num_prefill_tokens] = out
                else:
                    output = out
            else:
                # prefix-enabled attention
                output[:num_prefill_tokens] = self.forward_prefix(
                    query,
                    key,
                    value,
                    self.kv_cache_dtype,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.query_start_loc,
                    prefill_meta.seq_lens_tensor,
                    prefill_meta.context_lens_tensor,
                    prefill_meta.max_query_len,
                    self.alibi_slopes,
                    self.sliding_window[0],
                    layer._k_scale,
                    layer._v_scale,
                )

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            decode_output = self.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                (
                    decode_meta.block_tables
                    if self.attn_type != AttentionType.ENCODER_DECODER
                    else decode_meta.cross_block_tables
                ),
                (
                    decode_meta.seq_lens_tensor
                    if self.attn_type != AttentionType.ENCODER_DECODER
                    else decode_meta.encoder_seq_lens_tensor
                ),
                (
                    decode_meta.max_decode_seq_len
                    if self.attn_type != AttentionType.ENCODER_DECODER
                    else decode_meta.max_encoder_seq_len
                ),
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                layer._k_scale,
                layer._v_scale,
            )
            # print(decode_output)
            output[num_prefill_tokens:] = decode_output

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)

    @staticmethod
    def forward_prefix(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache_dtype: str,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_tensor: torch.Tensor,
        context_lens: torch.Tensor,
        max_query_len: int,
        alibi_slopes: Optional[torch.Tensor],
        sliding_window: Optional[int],
        k_scale: float,
        v_scale: float,
    ) -> torch.Tensor:
        output = torch.empty_like(query)
        context_attention_fwd(
            query,
            key,
            value,
            output,
            kv_cache_dtype,
            key_cache,
            value_cache,
            block_tables,
            # query_start_loc is (batch_size + 1,)
            query_start_loc[:-1],
            seq_lens_tensor,
            context_lens,
            max_query_len,
            k_scale,
            v_scale,
            alibi_slopes,
            sliding_window,
        )
        return output

    def forward_decode(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        k_scale: float,
        v_scale: float,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> torch.Tensor:
        if blocksparse_vert_stride is not None and blocksparse_vert_stride > 1:
            # use blocksparse paged attention
            block_size = value_cache.size(-1)
            assert (
                blocksparse_block_size > 0 and blocksparse_block_size % block_size == 0
            ), (
                f"{blocksparse_block_size=} needs to be a multiple of"
                f"{block_size=} used in block_tables."
            )

        output = torch.empty_like(query)
        block_size = value_cache.shape[3]
        num_seqs, num_heads, head_size = query.shape

        batch_size = num_seqs
        num_query_heads = num_heads
        num_queries_per_kv = num_query_heads // num_kv_heads

        # TODO
        # NOTE(ngl): Since vLLM uses cuda graph for decode as default,
        #  and the example workloads for the cuda graph capture are all >128,
        #  it will always use 3d, currently.
        use_3d = max_seq_len > 128
        # print(f"use 3d triton paged attention: {use_3d}")

        if not use_3d:
            paged_attention_2d(
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
                batch_size,
                num_query_heads,
                num_queries_per_kv,
                head_size,
            )
        else:
            paged_attention_3d(
                output,
                query,
                key_cache,
                value_cache,
                scale,
                block_tables,
                seq_lens,
                alibi_slopes,
                block_size,
                batch_size,
                num_query_heads,
                num_queries_per_kv,
                head_size,
            )

        return output
