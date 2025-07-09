"""Custom normalization layers."""
from typing import Optional, Tuple, Union, NamedTuple

import os
import time
import torch

import triton
import triton.language as tl
from triton import Config
import triton_dejavu
from triton_dejavu.utils import unpack_grid, get_random_key, global_metadata_store


def fallback_heuristic(key):
    ret = Config({'BLOCK_N_SIZE': 1024 if key[1] <= 128 else 4096}, num_warps=16, num_stages=4)
    return ret

def informed_fallback_next(key, cache):
    ret = cache[min(cache.keys(), key=lambda x: abs(x - key[7]))]
    return ret

def prepare_informed_fallback(cache):
    ret = {int(k[7]): c for k, c in cache.items()}
    return ret

def fused_informed_fallback_next(key, cache):
    ret = cache[min(cache.keys(), key=lambda x: abs(x - key[4]))]
    return ret

def fused_prepare_informed_fallback(cache):
    ret = {int(k[4]): c for k, c in cache.items()}
    return ret

# lazy functions for paper evals
use_bo = lambda: os.getenv('NGL_EXP_USE_BO', '0') == '1'
use_random = lambda: os.getenv('NGL_EXP_USE_RANDOM_SEARCH', '0') == '1'
bo_time = lambda: int(os.getenv('NGL_EXP_BO_TIME', '360'))


def _select_informed_fallback():
    fallback_mode = os.getenv('NGL_EXP_FALLBACK', 'none')
    if fallback_mode == 'static':
        return None, None
    if fallback_mode == 'next':
        return informed_fallback_next, prepare_informed_fallback
    return informed_fallback_next, prepare_informed_fallback

# defaults to work without env
select_fallback_heuristic = lambda: fallback_heuristic if os.getenv('NGL_EXP_FALLBACK', 'none') == 'static' else None
select_informed_fallback = lambda: _select_informed_fallback()[0]
select_prepare_informed_fallback = lambda: _select_informed_fallback()[1]


def _select_fused_informed_fallback():
    fallback_mode = os.getenv('NGL_EXP_FALLBACK', 'none')
    if fallback_mode == 'static':
        return None, None
    if fallback_mode == 'next':
        return fused_informed_fallback_next, fused_prepare_informed_fallback
    return fused_informed_fallback_next, fused_prepare_informed_fallback

# defaults to work without env
fused_select_informed_fallback = lambda: _select_fused_informed_fallback()[0]
fused_select_prepare_informed_fallback = lambda: _select_fused_informed_fallback()[1]


gpu_name = torch.cuda.get_device_name()


_my_autotune_metadata_key = get_random_key(prefix="rms_norm")

def metadata_fn(
    grid: tuple,
    metadata: NamedTuple,
    args: dict,
):
    grid_x, grid_y, grid_z = unpack_grid(grid)
    num_warps = metadata.num_warps
    num_stages = metadata.num_stages
    cluster_x, cluster_y, cluster_z = metadata.cluster_dims
    shared_memory = metadata.shared
    autotune_metadata = global_metadata_store.get(_my_autotune_metadata_key, "<no-autotune>")
    # args just contains NON-CONSTANT arguments
    num_tokens, hidden_size = args["x_ptr"].shape
    # num tokens are treated as batch
    dtype_size = args["x_ptr"].element_size()
    return {
        "name":
        # f"rmsnorm_triton_<grid:{grid_x}x{grid_y}x{grid_z}>_<cluster:{cluster_x}x{cluster_y}x{cluster_z}>_<warps:{num_warps}>_<shared:{shared_memory}>_<stages:{num_stages}>",
        f"rmsnorm_triton_<grid:{grid_x}x{grid_y}x{grid_z}>_<cluster:{cluster_x}x{cluster_y}x{cluster_z}>_<shared:{shared_memory}>_{autotune_metadata}",
        "flops16": 4 * num_tokens * hidden_size,
        "bytes": dtype_size * num_tokens * hidden_size * 3
    }


# from https://github.com/ELS-RD/kernl/blob/main/experimental/llama-v2/kernel/fused_kernel_ff.py
@triton_dejavu.autotune(
    config_space=triton_dejavu.ConfigSpace(
        {'BLOCK_N_SIZE': [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]},
        num_warps=[4, 8, 16, 32],
        num_stages=[1, 2, 4, 6, 8, 10, 12],
        num_ctas=[1],
    ),
    # TODO batch size and sequence length is part of grid...
    key=['stride_x_batch', 'stride_x_m', 'stride_x_k', 'stride_rms_w', 
         'stride_out_batch', 'stride_out_m', 'stride_out_k',
         'N_SIZE'
        ], 
    use_cuda_graph=True,
    # fallback_heuristic = fallback_heuristic,
    # informed_fallback = informed_fallback_next,
    # prepare_informed_fallback = prepare_informed_fallback,
    fallback_heuristic = select_fallback_heuristic(),
    informed_fallback = select_informed_fallback(),
    prepare_informed_fallback = select_prepare_informed_fallback(),
    use_bo=use_bo(),
    use_random_search=use_random(),
    search_max_search_t=bo_time(),
    search_max_share=1.0,  # will anyhow timeout...
    metadata_key=_my_autotune_metadata_key,
)
@triton.jit(launch_metadata=metadata_fn)
def rmsnorm_triton(x_ptr, rms_w_ptr, output_ptr,
                   stride_x_batch, stride_x_m, stride_x_k,
                   stride_rms_w,
                   stride_out_batch, stride_out_m, stride_out_k,
                   N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_N = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        # var += tl.math.pow(x.to(tl.float32), 2)
        # var += x * x
        var += (x.to(tl.float32) * x.to(tl.float32))

    var = tl.sum(var, axis=0) / N_SIZE
    # rstd = tl.math.rsqrt(var + eps)
    rstd = 1/tl.math.sqrt(var + eps)

    # multiply by weight and add bias
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        rms_w = tl.load(rms_w_ptr + offs_n * stride_rms_w, mask=x_ptr_mask)

        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        out = x_hat * rms_w
        out_off = pid_batch * stride_out_batch + pid_m * stride_out_m + offs_n * stride_out_k
        tl.store(output_ptr + out_off, out, mask=x_ptr_mask)


def rmsnorm_triton_wrapper(x, rms_w, eps=1e-6):
    out = torch.empty_like(x)
    if len(x.shape) == 3:
        batch, M, K = x.shape
        stride_x_batch, stride_x_m, stride_x_k = x.stride()
        # stride_rms_w = rms_w.stride()
        stride_rms_w = rms_w.stride()[0]
        stride_out_batch, stride_out_m, stride_out_k = out.stride()
    else:
        batch, K = x.shape
        M = 1
        stride_x_batch, stride_x_k = x.stride()
        stride_x_m = 1
        # stride_rms_w = rms_w.stride()
        stride_rms_w = rms_w.stride()[0]
        stride_out_batch, stride_out_k = out.stride()
        stride_out_m = 1
    assert rms_w.shape[-1] == K
        
    rmsnorm_triton[(batch, M,)](x, rms_w, out,
                                stride_x_batch, stride_x_m, stride_x_k,
                                stride_rms_w,
                                stride_out_batch, stride_out_m, stride_out_k,
                                eps=eps,
                                N_SIZE=K, 
                                # BLOCK_N_SIZE=1024,
                                )
    return out




_my_autotune_metadata_key_fused = get_random_key(prefix="fused_rms_norm")

def metadata_fn_fused(
    grid: tuple,
    metadata: NamedTuple,
    args: dict,
):
    grid_x, grid_y, grid_z = unpack_grid(grid)
    num_warps = metadata.num_warps
    num_stages = metadata.num_stages
    cluster_x, cluster_y, cluster_z = metadata.cluster_dims
    shared_memory = metadata.shared
    autotune_metadata = global_metadata_store.get(_my_autotune_metadata_key_fused, "<no-autotune>")
    # args just contains NON-CONSTANT arguments
    num_tokens, hidden_size = args["x_ptr"].shape
    # num tokens are treated as batch
    dtype_size = args["x_ptr"].element_size()
    return {
        "name":
        # f"rmsnorm_triton_<grid:{grid_x}x{grid_y}x{grid_z}>_<cluster:{cluster_x}x{cluster_y}x{cluster_z}>_<warps:{num_warps}>_<shared:{shared_memory}>_<stages:{num_stages}>",
        f"rmsnorm_triton_<grid:{grid_x}x{grid_y}x{grid_z}>_<cluster:{cluster_x}x{cluster_y}x{cluster_z}>_<shared:{shared_memory}>_{autotune_metadata}",
        "flops16": 4 * num_tokens * hidden_size,
        "bytes": dtype_size * num_tokens * hidden_size * 3
    }

@triton_dejavu.autotune(
    config_space=triton_dejavu.ConfigSpace(
        {'BLOCK_N_SIZE': [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]},
        num_warps=[4, 8, 16, 32],
        num_stages=[1, 2, 4, 6, 8, 10, 12],
        num_ctas=[1],
    ),
    # TODO batch size and sequence length is part of grid...
    key=[
        'stride_x_batch', 'stride_x_m', 'stride_x_k', 'stride_rms_w',
        'N_SIZE'
        ],
    restore_value = ['residual_ptr', 'x_ptr'],
    use_cuda_graph=True,
    # fallback_heuristic = fallback_heuristic,
    # informed_fallback = fused_informed_fallback_next,
    # prepare_informed_fallback = fused_prepare_informed_fallback,
    fallback_heuristic = select_fallback_heuristic(),
    informed_fallback = fused_select_informed_fallback(),
    prepare_informed_fallback = fused_select_prepare_informed_fallback(),
    use_bo=use_bo(),
    use_random_search=use_random(),
    search_max_search_t=bo_time(),
    search_max_share=1.0,  # will anyhow timeout...
    metadata_key=_my_autotune_metadata_key_fused,
)
@triton.jit(launch_metadata=metadata_fn_fused)
def fused_add_rmsnorm_triton(x_ptr, rms_w_ptr, residual_ptr,
                   stride_x_batch, stride_x_m, stride_x_k,
                   stride_rms_w,
                   N_SIZE: tl.constexpr, eps: tl.constexpr, 
                   BLOCK_N_SIZE: tl.constexpr, 
                   ):
    """
    changes x and residuals in place!
    """
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_N = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        r = tl.load(residual_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        x += r
        # var += tl.math.pow(x.to(tl.float32), 2)
        # var += x * x
        var += (x.to(tl.float32) * x.to(tl.float32))
        # tl.store(residual_ptr + offs_m + offs_n * stride_x_k, x, mask=x_ptr_mask, autotune_mask=autotune_mask)
        tl.store(residual_ptr + offs_m + offs_n * stride_x_k, x, mask=x_ptr_mask)

    var = tl.sum(var, axis=0) / N_SIZE
    # rstd = tl.math.rsqrt(var + eps)
    rstd = 1/tl.math.sqrt(var + eps)

    # multiply by weight and add bias
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        rms_w = tl.load(rms_w_ptr + offs_n * stride_rms_w, mask=x_ptr_mask)

        x = tl.load(residual_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        out = x_hat * rms_w
        out_off = pid_batch * stride_x_batch + pid_m * stride_x_m + offs_n * stride_x_k
        # tl.store(x_ptr + out_off, out, mask=x_ptr_mask, autotune_mask=autotune_mask)
        tl.store(x_ptr + out_off, out, mask=x_ptr_mask)


def fused_add_rmsnorm_triton_wrapper(x, residual, rms_w, eps=1e-6):
    if len(x.shape) == 3:
        batch, M, K = x.shape
        stride_x_batch, stride_x_m, stride_x_k = x.stride()
        stride_rms_w = rms_w.stride()[0]
    else:
        batch, K = x.shape
        M = 1
        stride_x_batch, stride_x_k = x.stride()
        stride_x_m = 1
        stride_rms_w = rms_w.stride()[0]
    # print(stride_rms_w, stride_x_batch, stride_x_k, stride_x_m)
    assert rms_w.shape[-1] == K
    assert x.shape == residual.shape
    # out = torch.empty_like(x)
        
    # changes x and residuals in place!
    fused_add_rmsnorm_triton[(batch, M,)](x, rms_w, residual,
                                          stride_x_batch, stride_x_m, stride_x_k,
                                          stride_rms_w,
                                          eps=eps,
                                          N_SIZE=K, 
                                          # BLOCK_N_SIZE=1024,
                                          )
    return x, residual

