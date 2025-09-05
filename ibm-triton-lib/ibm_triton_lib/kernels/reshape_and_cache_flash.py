import os
import torch
import triton
import triton.language as tl
from triton import Config
import triton_dejavu


_hopper_or_above = torch.cuda.get_device_capability("cuda")[0] >= 9 if torch.version.cuda else False

# def fallback_heuristic(key):
#     ret = Config(
#         {"TILE_SIZE": 1024 if key[1] <= 128 else 4096}, num_warps=16, num_stages=4
#     )
#     return ret
# 
# @triton_dejavu.autotune(
#     config_space=triton_dejavu.ConfigSpace(
#         {"TILE_SIZE": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]},
#         num_warps=[2, 4, 8, 16],
#         num_stages=[1, 2, 4, 6, 8, 10, 12],
#     ),
#     key=["key_stride", "value_stride", "head_size", "num_heads", "block_size",
#          "num_tokens"  # is quite sensitive to it?
#          ],
#     custom_data_storage=os.path.abspath(
#         os.path.join(os.path.dirname(__file__), "dejavu_data")
#     ),
#     use_cuda_graph=True,
#     # fallback_heuristic = fallback_heuristic,
#     use_bo=True,
#     search_max_search_t=520,
# )
@triton.jit
def reshape_and_cache_kernel_flash(
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    key_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
    value_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping_ptr,  # [num_tokens]
    k_scale,  # float32
    v_scale,  # float32
    key_stride: tl.int64,
    value_stride: tl.int64,
    block_stride: tl.int64,
    page_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):

    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        # Padding token that should be ignored.
        return

    tile_i = tl.program_id(axis=1)
    tile_offs = tl.arange(0, TILE_SIZE)
    tile_pos = tile_i * TILE_SIZE + tile_offs

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    src_key_idx = token_idx * key_stride
    src_value_idx = token_idx * value_stride

    tgt_idx = block_idx * block_stride + block_offset * page_stride

    # [TILE_SIZE]
    key_load = tl.load(
        key_ptr + src_key_idx + tile_pos,
        mask=tile_pos < (num_heads * head_size)
    )
    if key_load.dtype.is_fp8():
        key_tile = (key_load.to(tl.float32) * tl.load(k_scale)
                    ).to(tl.load(key_cache_ptr).dtype)
    else:
        key_tile = key_load

    # [TILE_SIZE]
    value_load = tl.load(
        value_ptr + src_value_idx + tile_pos,
        mask=tile_pos < (num_heads * head_size)
    )
    if value_load.dtype.is_fp8():
        value_tile = (value_load.to(tl.float32) * tl.load(v_scale)
                      ).to(tl.load(value_cache_ptr).dtype)
    else:
        value_tile = value_load

    tl.store(
        key_cache_ptr + tgt_idx + tile_pos,
        key_tile,
        mask=tile_pos < (num_heads * head_size),
    )
    tl.store(
        value_cache_ptr + tgt_idx + tile_pos,
        value_tile,
        mask=tile_pos < (num_heads * head_size),
    )
    return


def reshape_and_cache_flash(
    key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_heads, head_size]
    value_cache: torch.Tensor,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping: torch.Tensor,  # [num_tokens]
    k_scale: torch.Tensor,  # float32
    v_scale: torch.Tensor,  # float32
):
    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = key_cache.shape[1]
    n = num_heads * head_size

    key_stride = key.stride()[0]
    value_stride = key.stride()[0]
    block_stride = key_cache.stride()[0]
    page_stride = key_cache.stride()[1]

    head_stride = key_cache.stride()[2]
    assert head_stride == head_size, "only continous heads are supported"

    TILE_SIZE = min(2048, triton.next_power_of_2(n))
    if torch.version.hip:
        num_stages = 4
        num_warps = 8
    else: # cuda
        num_stages = 10
        num_warps = 16
        if not _hopper_or_above:
            TILE_SIZE = min(512, TILE_SIZE)

    # TODO: static launch grid?
    grid = lambda meta: (int(num_tokens), triton.cdiv(n, meta["TILE_SIZE"]))

    reshape_and_cache_kernel_flash[grid](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
        key_stride=key_stride,
        value_stride=value_stride,
        block_stride=block_stride,
        page_stride=page_stride,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        # autotune parameters
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return
