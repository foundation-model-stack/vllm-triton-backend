import torch


def is_fp8_dtype(t):
    return t == torch.float8_e4m3fn or t == torch.float8_e5m2


# NOTE: this might be slow... avoid calling repeatedly
def get_num_SMs(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


# Using rule based NUM_SPLITS computation
# L should be the minimal kv length in a batch
# BLOCK_L is the chosen block size
def compute_split_l(L, BLOCK_L, P=1, device=None):
    NUM_SMs = 132 if device is None else get_num_SMs(device)
    if P >= NUM_SMs:
        # there's already enough parallelism
        # no need to further split L
        return 1

    # Find minimum num_splits that will result in enough triton programs
    # TODO: does num_splits need to be power of 2?
    num_splits = 1
    split_size = L
    while (num_splits * P < NUM_SMs) and (split_size > BLOCK_L):
        num_splits *= 2
        split_size = L // num_splits

    return num_splits
