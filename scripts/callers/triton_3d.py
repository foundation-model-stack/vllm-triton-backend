import torch
from ibm_triton_lib.kernels import paged_attention_3d
from .base import DecodeCaller, PrefillCaller


class Triton3dAttentionDecodeCaller(DecodeCaller):
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
        kv_cache_dtype,  # unused
    ):
        num_query_heads = query.shape[1]
        num_kv_heads = key_cache.shape[1]
        block_size = key_cache.shape[3]
        num_queries_per_kv = num_query_heads // num_kv_heads
        max_num_blocks_per_seq = block_tables.shape[1]
        head_size = key_cache.shape[2]

        call_func_under_test = lambda: paged_attention_3d(
            output,
            query,
            key_cache,
            value_cache,
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


class Triton3dAttentionPrefillCaller(PrefillCaller):
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
        # with varlen
        # q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        # k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        # v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        # cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
        #    of the sequences in the batch, used to index into q.
        # cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
        #    of the sequences in the batch, used to index into kv.
        # max_seqlen_q: int. Maximum query sequence length in the batch.
        # max_seqlen_k: int. Maximum key sequence length in the batch.
        # out: (total, nheads, headdim).

        def call_and_process_output():
            prefill_flash_attention(
                q=query,
                k=key_cache,
                v=value_cache,
                causal=causal,
                sm_scale=softmax_scale,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                in_place_output=output,
                do_not_return_softmax_encodings=True,
            )

        return call_and_process_output
