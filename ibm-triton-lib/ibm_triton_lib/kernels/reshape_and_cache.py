
import os
import torch
import triton
import triton.language as tl
from triton import Config
import triton_dejavu


# not as lambda, for python3.9
def fallback_heuristic(key):
    ret = Config({'THREAD_BLOCK_SIZE': 1024 if key[1] <= 128 else 4096}, num_warps=16, num_stages=4)
    return ret

def informed_fallback_next(key, cache):
    ret = cache[min(cache.keys(), key=lambda x: abs(x - key[5]))]
    return ret

def prepare_informed_fallback(cache):
    ret = {int(k[5]): c for k, c in cache.items()}
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



@triton_dejavu.autotune(
    config_space=triton_dejavu.ConfigSpace(
        {'THREAD_BLOCK_SIZE': [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]},
        num_warps=[2, 4, 8, 16],
        num_stages=[1, 2, 4, 6, 8, 10, 12],
        num_ctas=[1],
        enable_warp_specialization=[False, True]
    ),
    # rep=10,
    # warmup=5,
    # key=['key_stride', 'value_stride', 'head_size', 'num_heads', 'block_size'], 
    key=['key_stride', 'value_stride', 'head_size', 'num_heads', 'block_size', 'num_tokens', 'x'], 
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
)
@triton.jit
def reshape_and_cache_kernel(
    key_ptr,            # [num_tokens, num_heads, head_size]
    value_ptr,          # [num_tokens, num_heads, head_size]
    key_cache_ptr,      # [num_blocks, num_heads, head_size/x, block_size, x]
    value_cache_ptr,    # [num_blocks, num_heads, head_size, block_size]
    slot_mapping_ptr,   # [num_tokens]
    key_stride: tl.constexpr,
    value_stride: tl.constexpr,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    # num_blocks: tl.constexpr,
    x: tl.constexpr,
    # n: tl.constexpr,
    THREAD_BLOCK_SIZE: tl.constexpr):

    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)

    # TODO: doesn't work in interpretated mode?
    if slot_idx < 0:
        # Padding token that should be ignored.
        return
        
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    
    thread_i = tl.program_id(axis=1)
    i = thread_i*THREAD_BLOCK_SIZE + tl.arange(0, THREAD_BLOCK_SIZE)
    
    src_key_idx = token_idx * key_stride + i
    src_value_idx = token_idx * value_stride + i

    head_idx = i // head_size
    head_offset = i % head_size
    x_idx = head_offset // x
    x_offset = head_offset % x

    tgt_key_idx = (block_idx * num_heads * (head_size//x) * block_size * x
                  + head_idx * (head_size // x) * block_size * x
                  + x_idx * block_size * x
                  + block_offset * x
                  + x_offset)
    tgt_value_idx = (block_idx * num_heads * head_size * block_size 
                    + head_idx * head_size * block_size 
                    + head_offset * block_size 
                    + block_offset)
    tgt_key = tl.load(key_ptr + src_key_idx, mask = src_key_idx < (num_tokens * key_stride))
    tgt_value = tl.load(value_ptr + src_value_idx, mask = src_value_idx < (num_tokens * value_stride))
    tl.store(key_cache_ptr + tgt_key_idx, tgt_key, mask = tgt_key_idx < (block_idx*block_size*num_heads*head_size + block_size*head_size*num_heads))
    tl.store(value_cache_ptr + tgt_value_idx, tgt_value, mask = tgt_value_idx < (block_idx*block_size*num_heads*head_size + block_size*head_size*num_heads))
    return


def reshape_and_cache(
        key: torch.Tensor,          # [num_tokens, num_heads, head_size]
        value: torch.Tensor,        # [num_tokens, num_heads, head_size]
        key_cache: torch.Tensor,    # [num_blocks, num_heads, head_size/x, block_size, x]
        value_cache: torch.Tensor,  # [num_blocks, num_heads, head_size, block_size]
        slot_mapping: torch.Tensor  # [num_tokens]
        ):
    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = key_cache.shape[3]
    if len(key_cache.shape) == 5:
        x = key_cache.shape[4]
    else:
        # for the reshape_and_cache_flash kernel
        x = 1
    n = num_heads * head_size
    # num_blocks = key_cache.shape[0]

    key_stride = key.stride()[0]
    value_stride = key.stride()[0]
        
    grid = lambda meta: (int(num_tokens), triton.cdiv(n, meta['THREAD_BLOCK_SIZE'])) 
    # print(f'expected grid: {num_tokens}, {triton.cdiv(n, 256)};\n' \
    #       f'key shape: {key.shape}; value shape: {value.shape};\n' \
    #       f'key cache shape: {key_cache.shape}; value cache shape: {value_cache.shape}\n' \
    #       f'key stride: {key.stride()}; value stride: {value.stride()}\n'\
    #       f'key cache stride: {key_cache.stride()}; value cache stride: {value_cache.stride()}\n')

    reshape_and_cache_kernel[grid](key, value, key_cache, value_cache, slot_mapping,
                                   key_stride, value_stride, num_tokens, num_heads, 
                                   head_size, block_size, 
                                   # num_blocks, 
                                   x, 
                                   # n,
                                   # THREAD_BLOCK_SIZE=256,
                                   )

    return
