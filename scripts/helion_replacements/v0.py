from __future__ import annotations

import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
from torch._inductor.runtime.triton_compat import libdevice
from helion.runtime import default_launcher as _default_launcher

@triton.jit
def _helion_kernel_helion_v0_attention(t_seq_lens, t_query_start_lens, t_query, t_block_tables, t_key_cache, t_value_cache, t_output, scale, _RDIM_SIZE_4: tl.constexpr, _RDIM_SIZE_6: tl.constexpr, _BLOCK_SIZE_5: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_3: tl.constexpr):
    # src[helion_unified_attention.py:43]: for seq_idx, kv_head_idx in hl.grid([num_seqs, num_kv_heads]):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    offset_0 = pid_0
    offset_1 = pid_1
    indices_15 = tl.arange(0, _RDIM_SIZE_4).to(tl.int32)
    indices_16 = tl.arange(0, _RDIM_SIZE_6).to(tl.int32)
    # src[helion_unified_attention.py:44]: seq_len = t_seq_lens[seq_idx]
    seq_len = tl.load(t_seq_lens + offset_0 * 1, None)
    # src[helion_unified_attention.py:45]: query_start = t_query_start_lens[seq_idx]
    query_start = tl.load(t_query_start_lens + offset_0 * 1, None)
    # src[helion_unified_attention.py:46]: query_end = t_query_start_lens[seq_idx + 1]
    add = 1 + offset_0
    query_end = tl.load(t_query_start_lens + add * 1, None)
    # src[helion_unified_attention.py:47]: query_len = query_end - query_start
    v_0 = query_end - query_start
    # src[helion_unified_attention.py:48]: context_len = seq_len - query_len
    v_1 = seq_len - v_0
    # src[helion_unified_attention.py:50]: for tile_q in hl.tile(query_start, query_end, block_size=None):
    # src[helion_unified_attention.py:51]:     for tile_m in hl.tile(kv_head_idx * num_queries_per_kv, (kv_head_idx+1)*num_queries_per_kv,
    # src[helion_unified_attention.py:52]:                       block_size=num_queries_per_kv):
    # src[helion_unified_attention.py:50-105]: ...
    end_0 = query_end.to(tl.int32)
    for offset_13 in tl.range(query_start.to(tl.int32), query_end.to(tl.int32), _BLOCK_SIZE_2):
        indices_13 = offset_13 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        mask_2 = indices_13 < query_end
        v_1_copy = v_1
        seq_len_copy = seq_len
        v_1_copy_0 = v_1_copy
        seq_len_copy_0 = seq_len_copy
        # src[helion_unified_attention.py:51]: for tile_m in hl.tile(kv_head_idx * num_queries_per_kv, (kv_head_idx+1)*num_queries_per_kv,
        mul = 4 * offset_1
        add_1 = 1 + offset_1
        mul_1 = 4 + 4 * offset_1
        # src[helion_unified_attention.py:51]: for tile_m in hl.tile(kv_head_idx * num_queries_per_kv, (kv_head_idx+1)*num_queries_per_kv,
        # src[helion_unified_attention.py:52]:                   block_size=num_queries_per_kv):
        # src[helion_unified_attention.py:53]:     block_m_size = tile_m.block_size * tile_q.block_size
        # src[helion_unified_attention.py:51-105]: ...
        for offset_12 in tl.range(mul.to(tl.int32), mul_1.to(tl.int32), _BLOCK_SIZE_3):
            indices_12 = offset_12 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
            v_1_copy_0_copy = v_1_copy_0
            seq_len_copy_0_copy = seq_len_copy_0
            v_1_copy_0_copy_0 = v_1_copy_0_copy
            seq_len_copy_0_copy_0 = seq_len_copy_0_copy
            # src[helion_unified_attention.py:53]: block_m_size = tile_m.block_size * tile_q.block_size
            mul_2 = _BLOCK_SIZE_2 * _BLOCK_SIZE_3
            # src[helion_unified_attention.py:55]: q = t_query[tile_q, tile_m, :].view([block_m_size, head_size])
            load = tl.load(t_query + (indices_13[:, None, None] * 4096 + indices_12[None, :, None] * 128 + indices_15[None, None, :] * 1), mask_2[:, None, None], other=0)
            q = tl.reshape(load, [_BLOCK_SIZE_2 * _BLOCK_SIZE_3, 128])
            # src[helion_unified_attention.py:58]: m = hl.full([block_m_size], float("-inf"), dtype=torch.float32) # device=q.device)
            m = tl.full([_BLOCK_SIZE_2 * _BLOCK_SIZE_3], float('-inf'), tl.float32)
            # src[helion_unified_attention.py:60]: l = hl.full([block_m_size], 1.0, dtype=torch.float32)
            full_1 = tl.full([_BLOCK_SIZE_2 * _BLOCK_SIZE_3], 1.0, tl.float32)
            # src[helion_unified_attention.py:62]: acc = hl.zeros([block_m_size, head_size], dtype=torch.float32)  # , device=q.device)
            acc = tl.full([_BLOCK_SIZE_2 * _BLOCK_SIZE_3, 128], 0.0, tl.float32)
            # src[helion_unified_attention.py:66]: max_seq_prefix_len = context_len + tile_q.end + (tile_m.block_size - 1) // num_queries_per_kv + 1
            tile_end = tl.minimum(offset_13 + _BLOCK_SIZE_2, end_0)
            v_2 = tl.cast(v_1_copy_0_copy_0, tl.int64)
            v_3 = v_2 + tile_end
            sub = -1 + _BLOCK_SIZE_3
            floordiv = triton_helpers.div_floor_integer(-1 + _BLOCK_SIZE_3, 4)
            v_4 = tl.cast(v_3, tl.int64)
            v_5 = v_4 + floordiv
            v_6 = tl.full([], 1, tl.int32)
            v_7 = v_5 + v_6
            # src[helion_unified_attention.py:67]: max_seq_prefix_len = torch.minimum(max_seq_prefix_len, seq_len)
            v_8 = triton_helpers.minimum(v_7, seq_len_copy_0_copy_0)
            # src[helion_unified_attention.py:68]: num_blocks = torch.ceil(max_seq_prefix_len / page_size)
            v_9 = tl.cast(v_8, tl.float32)
            v_10 = 0.0625
            v_11 = v_9 * v_10
            v_12 = libdevice.ceil(v_11)
            # src[helion_unified_attention.py:69]: for tile_n in hl.tile(num_blocks, block_size=None):
            # src[helion_unified_attention.py:70]:     block_n_size = tile_n.block_size * page_size
            # src[helion_unified_attention.py:71]:     blk_idxs = t_block_tables[seq_idx, tile_n].view(-1)
            # src[helion_unified_attention.py:69-101]: ...
            for offset_14 in tl.range(0, v_12.to(tl.int32)):
                indices_14 = offset_14 + tl.arange(0, 1).to(tl.int32)
                q_copy = q
                m_copy = m
                acc_copy = acc
                full_1_copy = full_1
                q_copy_0 = q_copy
                m_copy_0 = m_copy
                acc_copy_0 = acc_copy
                full_1_copy_0 = full_1_copy
                # src[helion_unified_attention.py:70]: block_n_size = tile_n.block_size * page_size
                mul_3 = 16 * _BLOCK_SIZE_5
                # src[helion_unified_attention.py:71]: blk_idxs = t_block_tables[seq_idx, tile_n].view(-1)
                load_1 = tl.load(t_block_tables + (offset_0 * 4 + indices_14 * 1), None)
                blk_idxs = tl.reshape(load_1, [_BLOCK_SIZE_5])
                # src[helion_unified_attention.py:73]: k = t_key_cache[blk_idxs, :, kv_head_idx, :].squeeze(2)
                load_2 = tl.load(t_key_cache + (blk_idxs[:, None, None] * 16384 + indices_16[None, :, None] * 1024 + offset_1 * 128 + indices_15[None, None, :] * 1), None)
                k = tl.reshape(load_2, [_BLOCK_SIZE_5, 16, 128])
                # src[helion_unified_attention.py:75]: v = t_value_cache[blk_idxs, :, kv_head_idx, :]
                v = tl.load(t_value_cache + (blk_idxs[:, None, None] * 16384 + indices_16[None, :, None] * 1024 + offset_1 * 128 + indices_15[None, None, :] * 1), None)
                # src[helion_unified_attention.py:77]: k = k.view([block_n_size, head_size]).transpose(0, 1)
                view_1 = tl.reshape(k, [16 * _BLOCK_SIZE_5, 128])
                k_1 = tl.permute(view_1, [1, 0])
                # src[helion_unified_attention.py:79]: qk = torch.mm(q, k) * scale
                mm = tl.dot(tl.cast(q_copy_0, tl.float16), tl.cast(k_1, tl.float16), input_precision='ieee', out_dtype=tl.float32)
                v_13 = tl.cast(scale, tl.float16)
                v_14 = mm * v_13
                # src[helion_unified_attention.py:86]: m_j = torch.maximum(m, torch.amax(qk, 1))
                amax = tl.cast(tl.max(v_14, 1), tl.float16)
                v_15 = tl.cast(amax, tl.float32)
                v_16 = triton_helpers.maximum(m_copy_0, v_15)
                # src[helion_unified_attention.py:88]: p = torch.exp(qk - m_j[:, None])
                subscript = v_16[:, None]
                v_17 = tl.cast(v_14, tl.float32)
                v_18 = v_17 - subscript
                v_19 = tl_math.exp(v_18)
                # src[helion_unified_attention.py:90]: l_j = torch.sum(p, 1)
                l_j = tl.cast(tl.sum(v_19, 1), tl.float32)
                # src[helion_unified_attention.py:92]: alpha = torch.exp(m - m_j)
                v_20 = m_copy_0 - v_16
                v_21 = tl_math.exp(v_20)
                # src[helion_unified_attention.py:94]: acc *= alpha[:, None]
                subscript_1 = v_21[:, None]
                v_22 = acc_copy_0 * subscript_1
                # src[helion_unified_attention.py:95]: l *= alpha + l_j
                v_23 = v_21 + l_j
                full_1 = full_1_copy_0 * v_23
                # src[helion_unified_attention.py:96]: m = m_j
                m = v_16
                # src[helion_unified_attention.py:99]: v_view = v.view([tile_n.block_size * page_size, head_size])
                v_view = tl.reshape(v, [16 * _BLOCK_SIZE_5, 128])
                # src[helion_unified_attention.py:101]: acc += torch.mm(p.to(v.dtype), v_view)
                v_25 = tl.cast(v_19, tl.float16)
                mm_1 = tl.dot(tl.cast(v_25, tl.float16), tl.cast(v_view, tl.float16), input_precision='ieee', out_dtype=tl.float32)
                v_26 = tl.cast(mm_1, tl.float32)
                acc = v_22 + v_26
            # src[helion_unified_attention.py:104]: acc = acc / l[:, None]
            subscript_2 = full_1[:, None]
            v_28 = acc / subscript_2
            # src[helion_unified_attention.py:105]: t_output[tile_q, tile_m, :] = acc.view([tile_q.block_size, tile_m.block_size, head_size])
            view_2 = tl.reshape(v_28, [_BLOCK_SIZE_2, _BLOCK_SIZE_3, 128])
            v_29 = tl.cast(view_2, tl.float16)
            tl.store(t_output + (indices_13[:, None, None] * 4096 + indices_12[None, :, None] * 128 + indices_15[None, None, :] * 1), v_29, mask_2[:, None, None])

def kernel_helion_v0_attention(t_output, t_query, t_key_cache, t_value_cache, t_block_tables, t_seq_lens, scale, t_query_start_lens, num_seqs, *, _launcher=_default_launcher):
    # src[helion_unified_attention.py:43]: for seq_idx, kv_head_idx in hl.grid([num_seqs, num_kv_heads]):
    _RDIM_SIZE_4 = 128
    _RDIM_SIZE_6 = 16
    _BLOCK_SIZE_5 = 1
    # src[helion_unified_attention.py:50]: for tile_q in hl.tile(query_start, query_end, block_size=None):
    # src[helion_unified_attention.py:51]:     for tile_m in hl.tile(kv_head_idx * num_queries_per_kv, (kv_head_idx+1)*num_queries_per_kv,
    # src[helion_unified_attention.py:52]:                       block_size=num_queries_per_kv):
    # src[helion_unified_attention.py:50-105]: ...
    _BLOCK_SIZE_2 = 32
    # src[helion_unified_attention.py:51]: for tile_m in hl.tile(kv_head_idx * num_queries_per_kv, (kv_head_idx+1)*num_queries_per_kv,
    # src[helion_unified_attention.py:52]:                   block_size=num_queries_per_kv):
    # src[helion_unified_attention.py:53]:     block_m_size = tile_m.block_size * tile_q.block_size
    # src[helion_unified_attention.py:51-105]: ...
    _BLOCK_SIZE_3 = 4
    # src[helion_unified_attention.py:43]: for seq_idx, kv_head_idx in hl.grid([num_seqs, num_kv_heads]):
    # src[helion_unified_attention.py:44]:     seq_len = t_seq_lens[seq_idx]
    # src[helion_unified_attention.py:45]:     query_start = t_query_start_lens[seq_idx]
    # src[helion_unified_attention.py:43-105]: ...
    _RDIM_SIZE_7 = triton.next_power_of_2(16 * _BLOCK_SIZE_5)
    _launcher(_helion_kernel_helion_v0_attention, (num_seqs, 8), t_seq_lens, t_query_start_lens, t_query, t_block_tables, t_key_cache, t_value_cache, t_output, scale, _RDIM_SIZE_4, _RDIM_SIZE_6, _BLOCK_SIZE_5, _BLOCK_SIZE_2, _BLOCK_SIZE_3, num_warps=8, num_stages=1)