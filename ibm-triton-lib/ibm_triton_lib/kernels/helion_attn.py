

import torch

import helion
import helion.language as hl

import math

@helion.kernel
def kernel_helion_v0_attention(
    t_output,  # [num_tokens, num_query_heads, head_size]
    t_query,  # [num_tokens, num_query_heads, head_size]
    t_key_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_value_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_block_tables,  # [num_seqs, max_num_blocks_per_seq]
    t_seq_lens,  # [num_seqs]
    scale,
    # k_scale,
    # v_scale,
    # num_query_heads,
    # num_queries_per_kv,
    t_query_start_lens, # [num_seqs+1]
    num_seqs,
    #  head_size,
    #  head_size_padded,
):
    # seq_idx etc. is determined by helion?
    head_size = hl.specialize(t_query.size(2))
    num_kv_heads = hl.specialize(t_key_cache.size(2))
    num_query_heads = hl.specialize(t_query.size(1))
    page_size = hl.specialize(t_value_cache.size(1))
    # q_max_range = hl.specialize(t_query.size(0) * num_query_heads)
    num_queries_per_kv = num_query_heads // num_kv_heads

    for seq_idx, kv_head_idx in hl.grid([num_seqs, num_kv_heads]):
        seq_len = t_seq_lens[seq_idx]
        query_start = t_query_start_lens[seq_idx]
        query_end = t_query_start_lens[seq_idx + 1]
        query_len = query_end - query_start
        context_len = seq_len - query_len
        # TODO: enable tuning again
        for tile_q in hl.tile([query_len, num_query_heads, head_size], block_size=[page_size//num_queries_per_kv, num_queries_per_kv, head_size]):
            q = t_query[tile_q]
            m = hl.full([tile_q[0], tile_q[1]], float("-inf"), dtype=torch.float32)
            l = hl.full([tile_q[0], tile_q[1]], 1.0, dtype=torch.float32)
            acc = hl.zeros(tile_q, dtype=torch.float32)
            # no [] for 1d...here! 
            # for tile_b in hl.tile(math.ceil(num_seqs/page_size), block_size=None):
            # TODO: make block size depending on tile_q
            for tile_b in hl.tile(math.ceil(num_seqs/page_size), block_size=1):
            # for tile_b in hl.tile([seq_idx, 0], [seq_idx, math.ceil(num_seqs/block_size)], block_size=[1, None]):
                # k = t_key_cache[tile_b + t_block_tables[(seq_idx, )+tile_b], :, kv_head_idx, :]
                # v = t_value_cache[tile_b + t_block_tables[seq_idx+tile_b], :, kv_head_idx, :]
                blk_idxs = t_block_tables[seq_idx, tile_b]
                # blk_idxs = t_block_tables[tile_b]
                k = t_key_cache[blk_idxs, :, kv_head_idx, :]
                # k = torch.take(t_key_cache[:, :, kv_head_idx, :], torch.broadcast_to(blk_idxs[0, :], t_key_cache.shape()), 0)
                # k = torch.take_along_dim(t_key_cache[:, :, kv_head_idx, :], torch.broadcast_to(blk_idxs[0, :], t_key_cache.shape()), 0)
                # k = torch.take_along_dim(t_key_cache[:, :, kv_head_idx, :], blk_idxs.unsqueeze(1).unsqueeze(2).to(torch.long), 0)
                v = t_value_cache[blk_idxs, :, kv_head_idx, :]
                # qk = (q @ k) * scale
                # qk = torch.bmm(q, k.transpose(1, 2)) * scale
                q_view = q.reshape([-1, num_query_heads, head_size])
                k_view = k.reshape([-1, page_size, head_size]).transpose(1, 2)
                qk = torch.bmm(q_view, k_view) * scale
                m_j = torch.maximum(m, torch.amax(qk, 1))
                p = torch.exp2(qk - m_j[:, :, None])
                l_j = torch.sum(p, 1)
                alpha = torch.exp2(m - m_j)
                acc = acc * alpha[:, :, None]
                l = l * alpha + l_j
                m = m_j

                acc = acc + (p @ v)
            
            acc = acc / l[:, :, None]
            t_output[tile_q] = acc





        """"    
            for tile_b in hl.tile(t_block_tables[seq_idx].size(0), block_size=[None]):
                blk_idxs = t_block_tables[seq_idx, tile_b]
                blk_start = blk_idx * block_size
                blk_end = torch.minimum(blk_start + block_size, seq_len)
                torch.where




                for k_idx in hl.range(blk_start, blk_end):
                    k_head_idx = hl.arange(tile_q[1]) // num_queries_per_kv
                    k = t_key_cache[k_idx, k_head_idx, :]
                    qk = hl.matmul(q, k


        for q_idx in range(query_start, query_end):
            for q_head in range(num_query_heads):
                q = t_query[q_idx, q_head, :]  # [head_size]
                m_i = float("-inf")
                l_i = 0.0
                acc = hl.zeros([head_size], dtype=torch.float32)
                for blk_idx in range((seq_len + 31) // 32):
                    block_start = t_block_tables[seq_idx, blk_idx] * 32
                    block_end = min(block_start + 32, seq_len)
                    if block_start >= block_end:
                        continue
                    for k_idx in range(block_start, block_end):
                        k_head = q_head // num_queries_per_kv
                        k = t_key_cache[k_idx, k_head, :]  # [head_size]
                        qk = hl.dot(q, k) * scale  # scalar
                        m_ij = hl.maximum(m_i, qk)
                        qk = qk - m_ij
                        p = hl.exp2(qk)
                        l_ij = p
                        alpha = hl.exp2(m_i - m_ij)
                        l_i = l_i * alpha + l_ij
                        acc = acc * alpha + p * t_value_cache[k_idx, k_head, :]
                        m_i = m_ij
                m_i += hl.log2(l_i)
                acc = acc / l_i
                t_output[q_idx, q_head, :] = acc
        """

def helion_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    block_size = v.shape[1]
    assert (
        q.element_size() >= 2 or block_size >= 32
    ), "Block size must be at least 32 for fp8"

    use_alibi_slopes = alibi_slopes is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    # BLOCK_M = 32
    # BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    # total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    # grid = (q.shape[0] // (BLOCK_M // num_queries_per_kv) + num_seqs, num_kv_heads)

    kernel_helion_v0_attention(
        t_output=out,
        t_query=q,
        t_key_cache=k,
        t_value_cache=v,
        t_block_tables=block_table,
        t_seq_lens=seqused_k,
        scale=softmax_scale,
        # k_scale=k_descale,
        # v_scale=v_descale,
        t_query_start_lens=cu_seqlens_q,
        num_seqs=num_seqs,
    )


