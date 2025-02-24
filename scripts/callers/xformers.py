import torch
from xformers.ops import fmha
from .base import DecodeCaller


class XformersCaller(DecodeCaller):
    @staticmethod
    def make_call_func(
        output,
        query,
        key_cache,
        value_cache,
        num_seqs,
        seq_lens,
        max_seq_len,
        scale,  # unused
        block_tables,
        alibi_slopes,  # unused
        kv_cache_dtype,  # unused
    ):
        num_blocks = key_cache.shape[0]
        block_size = key_cache.shape[3]

        def transform_kv_cache(x):
            assert x.shape[0] == num_blocks
            assert x.shape[3] == block_size

            out = torch.empty(
                1, x.shape[0] * x.shape[3], x.shape[1], x.shape[2], dtype=x.dtype
            )

            for block_idx in range(x.shape[0]):
                for token_idx in range(x.shape[3]):
                    out[0, block_idx * x.shape[3] + token_idx, :, :] = x[
                        block_idx, :, :, token_idx
                    ]

            return out

        key_cache_xformers = transform_kv_cache(key_cache)
        value_cache_xformers = transform_kv_cache(value_cache)

        block_type = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask

        attn_bias = block_type.from_seqlens(
            q_seqlen=[1] * num_seqs,
            kv_padding=max_seq_len,
            kv_seqlen=seq_lens.tolist(),
        )

        make_paged_kwargs = {
            "paged_type": fmha.attn_bias.PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
        }

        attn_bias_paged = attn_bias.make_paged(
            block_tables=block_tables, page_size=block_size, **make_paged_kwargs
        )
        op = fmha.triton_splitk.FwOp
        op.BLOCK_N = block_size

        call_func_under_test = lambda: fmha.memory_efficient_attention_forward(
            query.view(1, query.shape[0], query.shape[1], query.shape[2]),
            key_cache_xformers,
            value_cache_xformers,
            attn_bias_paged,
            op=op,
        )

        return call_func_under_test

    @classmethod
    def select_output(cls, x, y):
        return y.view(y.shape[1], y.shape[2], y.shape[3])

    @staticmethod
    def requires_allocated_output() -> bool:
        return False
