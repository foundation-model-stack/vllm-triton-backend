import torch
from vllm import _custom_ops as ops
from .base import DecodeCaller


class VllmCudaV1Caller(DecodeCaller):
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
        num_kv_heads = key_cache.shape[1]
        block_size = key_cache.shape[3]

        # Using default kv_scale
        k_scale = v_scale = torch.ones(1, device=query.device)

        call_func_under_test = lambda: ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

        return call_func_under_test
