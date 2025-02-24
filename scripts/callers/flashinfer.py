import torch
import flashinfer
from .base import DecodeCaller


class FlashInferCaller(DecodeCaller):
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
        alibi_slopes,  # unused
        kv_cache_dtype,  # unused
    ):
        num_blocks = key_cache.shape[0]
        num_query_heads = query.shape[1]
        num_kv_heads = key_cache.shape[1]
        block_size = key_cache.shape[3]
        head_size = key_cache.shape[2]

        def transform_kv_cache(x):
            out = torch.transpose(x, 1, 3)
            out = torch.transpose(out, 2, 3)
            return out.contiguous()

        key_cache_flashinfer = transform_kv_cache(key_cache).unsqueeze(1)
        value_cache_flashinfer = transform_kv_cache(value_cache).unsqueeze(1)

        key_value_cache = torch.cat(
            (key_cache_flashinfer, value_cache_flashinfer), 1
        ).contiguous()

        kv_indptr = [0]
        kv_indices = []
        kv_last_page_lens = []
        for i in range(num_seqs):
            seq_len = seq_lens[i]
            assert seq_len > 0
            num_blocks = (seq_len + block_size - 1) // block_size
            kv_indices.extend(block_tables[i, :num_blocks])
            kv_indptr.append(kv_indptr[-1] + num_blocks)
            kv_last_page_len = seq_len % block_size
            if kv_last_page_len == 0:
                kv_last_page_len = block_size
            kv_last_page_lens.append(kv_last_page_len)

        kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
        kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
        kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

        workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer,
            "NHD",
            use_tensor_cores=((num_query_heads // num_kv_heads) > 4),
        )
        wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            num_query_heads,
            num_kv_heads,
            head_size,
            block_size,
            "NONE",
            data_type=query.dtype,
        )

        call_func_under_test = lambda: wrapper.forward(
            query, key_value_cache, logits_soft_cap=None
        )

        return call_func_under_test

    @staticmethod
    def requires_allocated_output() -> bool:
        return False
