import triton.language as tl
import triton

# Some of the functions in this file are adapted from Shawn Tan's stickbreaking-attention repo
# https://github.com/shawntan/stickbreaking-attention/blob/seq-aligned-folded/stickbreaking_attention/sb_varlen/softplus.py


def _generate_asm(num_pack):
    template = """
        .reg .pred p;
        setp.gt.f32  p, ${in_reg}, 15.;
        @p  mov.f32  ${out_reg}, ${in_reg};
        @!p ex2.approx.ftz.f32 ${out_reg}, ${in_reg};
        @!p add.f32            ${out_reg}, ${out_reg}, 1.0;
        @!p lg2.approx.ftz.f32 ${out_reg}, ${out_reg};
    """
    out_str = ""

    for i in range(num_pack):
        inner_str = template.format(out_reg=i, in_reg=i + num_pack)
        out_str += "{" + inner_str + "}\n"
    # flatten out because torch.compile doesn't like newlines
    out_str = " ".join(out_str.split("\n"))
    return out_str


def _generate_constraints(num_pack):
    return (
        ",".join("=r" for i in range(num_pack))
        + ","
        + ",".join("r" for i in range(num_pack))
    )


NUM_REG: tl.constexpr = 1
asm_str: tl.constexpr = _generate_asm(NUM_REG)
constraints_str: tl.constexpr = _generate_constraints(NUM_REG)


@triton.jit
def softplus(x, is_compiling: tl.constexpr = False):
    if is_compiling:
        tl.static_print("Using triton softplus.")
        out = tl.where(x < 15.0, tl.math.log2(1 + tl.math.exp2(x)), x)
        return out
    else:
        out = tl.inline_asm_elementwise(
            asm=asm_str,
            constraints=constraints_str,
            pack=NUM_REG,
            args=[
                x,
            ],
            dtype=tl.float32,
            is_pure=True,
        )
        return out


@triton.jit
def cumsum(x, block_range=None, USE_DOT_CUMSUM: tl.constexpr = False):
    if USE_DOT_CUMSUM:
        cm = tl.where(
            block_range[:, None] >= block_range[None, :], 1.0, 0.0
        )  # lower triangular matrix
        return tl.dot(x, cm)
    else:
        return tl.cumsum(x, axis=1, reverse=True)


@triton.jit
def get_split_tblocks_range(split_idx, kv_len, BLOCK_T, num_splits):
    num_tblocks = (kv_len + BLOCK_T - 1) // BLOCK_T
    tblock_start = (split_idx * num_tblocks) // num_splits
    tblock_end = ((split_idx + 1) * num_tblocks) // num_splits
    return tblock_start, tblock_end


@triton.jit
def attend_one_block(
    q,
    k,
    v,
    qk_scale,
    m_i,
    d_i,
    acc,
    alibi_slopes,  # [BLOCK_SS,]
    alibi_distances,  # [BLOCK_T,]
    IS_LAST_BLOCK,  # on_band == IS_LAST_BLOCK, dynamic
    tb_len_max,  # Number of tokens along page size (token) dimension. 0 < t_len <= BLOCK_T and it is a dynamic value
    offs_t: tl.constexpr,
    FORCE_FP16_PV: tl.constexpr,
    QUANTIZE_P: tl.constexpr,
    MAX_FP8: tl.constexpr,
    IS_STICKBREAKING: tl.constexpr,
    USE_DOT_CUMSUM: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    ATTEND_CURRENT: tl.constexpr,
):
    kv_len_dim: tl.constexpr = 1 if not TRANSPOSED else 0  # seqlen dimension

    # Compute logits
    if not TRANSPOSED:
        k = k.T  # [D, BLOCK_T]
        logits = tl.dot(q, k, out_dtype=tl.float32)  # [BLOCK_SS, BLOCK_T]
    else:
        q = q.T  # [D, BLOCK_SS]
        logits = tl.dot(k, q, out_dtype=tl.float32)  # [BLOCK_T, BLOCK_SS]

    logits *= qk_scale  # scale qk after mma

    if USE_ALIBI_SLOPES:
        alibi_biases = (
            alibi_slopes[:, None] * alibi_distances[None, :]
        )  # [BLOCK_SS, BLOCK_T]
        logits += alibi_biases if not TRANSPOSED else alibi_biases.T

    # Handle on band block special case
    t_mask = offs_t < tb_len_max
    if IS_LAST_BLOCK:
        if not IS_STICKBREAKING:
            t_mask_logits = t_mask[None, :] if not TRANSPOSED else t_mask[:, None]
            logits += tl.where(t_mask_logits, 0.0, float("-inf"))
        else:
            # v = tl.where(t_mask[:, None], v, 0.0)
            t_mask = offs_t < (tb_len_max if ATTEND_CURRENT else (tb_len_max - 1))

    if not IS_STICKBREAKING:  # regular softmax
        # -- compute scaling constant --
        m_i_new = tl.maximum(
            m_i, tl.max(logits, axis=kv_len_dim)
        )  # fp32, new max computation

        alpha = tl.math.exp2(m_i - m_i_new)  # fp32,  S4 (subtract new max from old max)
        p = tl.math.exp2(
            logits
            - (
                m_i_new[:, None]  # fp32, subtract current max # [BLOCK_SS, BLOCK_T]
                if not TRANSPOSED
                else m_i_new[None, :]
            )
        )

        # -- scale numerator ---
        acc *= alpha[:, None] if not TRANSPOSED else alpha[None, :]  # fp32 elmentwise
        # --- update m_i (max) and d_i (denominator) --
        m_i = m_i_new  # S2
        d_i = d_i * alpha + tl.sum(p, axis=kv_len_dim)  # S3
    else:  # stickbreaking attention
        # computations in log space
        log_om_beta = -softplus(
            logits,
        )  # [BLOCK_SS, BLOCK_T] or [BLOCK_T, BLOCK_SS]

        if TRANSPOSED:
            log_om_beta = log_om_beta.T  # [BLOCK_SS, BLOCK_T]
            logits = logits.T

        if IS_LAST_BLOCK:  # on_band
            log_om_beta = tl.where(t_mask[None, :], log_om_beta, 0.0)

        log_p = logits + d_i[:, None]  # [BLOCK_SS, BLOCK_T] # d_i is neg_log_acc
        d_i += tl.sum(log_om_beta, axis=1)  # [BLOCK_SS]
        log_p += cumsum(log_om_beta, block_range=offs_t, USE_DOT_CUMSUM=USE_DOT_CUMSUM)

        # non-log space
        p = tl.math.exp2(log_p)  # [BLOCK_SS, BLOCK_T]

        if IS_LAST_BLOCK:  # on_band
            p = tl.where(t_mask[None, :], p, 0.0)  # set masked elements to 0

        if TRANSPOSED:
            p = p.T  # [BLOCK_T, BLOCK_SS]

    p_scale = 1.0
    if FORCE_FP16_PV:
        # force fp16 for the 2nd bmm
        v = v.to(tl.float16)
    else:
        # align p with v.dtype for the 2nd bmm
        if QUANTIZE_P and v.dtype == tl.float8e4nv:
            tl.static_assert(
                not IS_STICKBREAKING
            )  # in stickbreaking p tensor values can become too small
            # --- dynamic quantization of p ---
            p_max = tl.max(tl.abs(p), axis=kv_len_dim, keep_dims=True)
            p_scale = p_max / MAX_FP8
            p_invs_scale = 1.0 / p_scale
            p = p * p_invs_scale  # fp32
    p = p.to(v.dtype)

    if not TRANSPOSED:
        acc += tl.dot(p, v, out_dtype=tl.float32) * p_scale  # [BLOCK_SS, D]
    else:
        acc += tl.dot(v.T, p, out_dtype=tl.float32) * p_scale  # [D, BLOCK_SS]

    return m_i, d_i, acc
