
import os
import torch
import triton
import triton.language as tl
from triton import Config
import triton_dejavu


def fallback_heuristic(key):
    ret = Config({'TILE_SIZE': 1024 if key[1] <= 128 else 4096}, num_warps=16, num_stages=4)
    return ret

@triton_dejavu.autotune(
    config_space=triton_dejavu.ConfigSpace(
        {'TILE_SIZE': [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]},
        num_warps=[2, 4, 8, 16],
        num_stages=[1, 2, 4, 6, 8, 10, 12],
    ),
    key=['key_stride', 'value_stride', 'head_size', 'num_heads', 'block_size'], 
    custom_data_storage=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "dejavu_data")
    ),
    use_cuda_graph=True,
    # fallback_heuristic = fallback_heuristic,
    use_bo=True,
    search_max_search_t=360,
)
@triton.jit
def reshape_and_cache_kernel_flash(
    key_ptr,            # [num_tokens, num_heads, head_size]
    value_ptr,          # [num_tokens, num_heads, head_size]
    key_cache_ptr,      # [num_blocks, num_heads, head_size, block_size]
    value_cache_ptr,    # [num_blocks, num_heads, head_size, block_size]
    slot_mapping_ptr,   # [num_tokens]
    num_tokens: tl.int64,
    key_stride: tl.int64,
    value_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    TILE_SIZE: tl.constexpr):

    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)

    if slot_idx < 0:
        # Padding token that should be ignored.
        return
        
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    
    prog_i = tl.program_id(axis=1)
    i = prog_i*TILE_SIZE + tl.arange(0, TILE_SIZE)
    
    src_key_idx = token_idx * key_stride + i
    src_value_idx = token_idx * value_stride + i

    head_idx = i // head_size
    head_offset = i % head_size

    tgt_key_idx = (block_idx * num_heads * head_size * block_size
                  + head_idx * head_size * block_size
                  + head_offset * block_size
                  + block_offset)
    tgt_value_idx = (block_idx * num_heads * head_size * block_size 
                    + head_idx * head_size * block_size 
                    + head_offset * block_size 
                    + block_offset)
    tgt_key = tl.load(key_ptr + src_key_idx, mask = src_key_idx < (num_tokens * key_stride))
    tgt_value = tl.load(value_ptr + src_value_idx, mask = src_value_idx < (num_tokens * value_stride))
    tl.store(key_cache_ptr + tgt_key_idx, tgt_key, mask = tgt_key_idx < (block_idx*block_size*num_heads*head_size + block_size*head_size*num_heads))
    tl.store(value_cache_ptr + tgt_value_idx, tgt_value, mask = tgt_value_idx < (block_idx*block_size*num_heads*head_size + block_size*head_size*num_heads))
    return


def reshape_and_cache_flash(
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
    n = num_heads * head_size

    key_stride = key.stride()[0]
    value_stride = key.stride()[0]

    # TODO: static launch grid

    grid = lambda meta: (int(num_tokens), triton.cdiv(n, meta['TILE_SIZE'])) 
    # print(f'expected grid: {num_tokens}, {triton.cdiv(n, 256)};\n' \
    #       f'key shape: {key.shape}; value shape: {value.shape};\n' \
    #       f'key cache shape: {key_cache.shape}; value cache shape: {value_cache.shape}\n' \
    #       f'key stride: {key.stride()}; value stride: {value.stride()}\n'\
    #       f'key cache stride: {key_cache.stride()}; value cache stride: {value_cache.stride()}\n')

    reshape_and_cache_kernel_flash[grid](
        key_ptr=key, 
        value_ptr=value, 
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping, 
        num_tokens=num_tokens,
        key_stride=key_stride, 
        value_stride=value_stride, 
        num_heads=num_heads, 
        head_size=head_size, 
        block_size=block_size, 
        )

    return
