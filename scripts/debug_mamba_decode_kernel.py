import torch
import vllm
# from vllm.model_executor.layers.mamba.mamba2_metadata import _query_start_loc_to_chunk_indices_offsets as query_start_loc_to_chunk_indices_offsets
# from vllm.model_executor.layers.mamba.ops.ssd_combined import mamba_chunk_scan_combined_varlen as mamba_chunk_scan_combined

from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update

from torch.profiler import profile, record_function, ProfilerActivity
from time import time

# generate dummy data for the test


def main(
    n_groups = 1,
    dstate = 128,
    nheads = 128, 
    headdim = 64, 
    # chunk_prefill_size = 2048,
    dtype = 'bfloat16',
    device = 'cuda',
    # chunk_size = 256,
    # change this to see different times
    batch_size = 256,
    has_initial_state: bool = True,
    repeats: int = 10,
    profile_on = False, 
):
    dtype = getattr(torch, dtype)

    def generate_dummy_data(batch_size):

        #derived parameters
        # seq_len = chunk_prefill_size + batch_size - 1

        hidden_states = torch.randn(batch_size, nheads, headdim, dtype=dtype, device=device)
        A = torch.rand(nheads, dtype=dtype, device=device)
        B = torch.randn(batch_size, dstate, dtype=dtype, device=device)
        C = torch.randn(batch_size, dstate, dtype=dtype, device=device)
        D = torch.randn(nheads, dtype=dtype, device=device)
        dt = torch.randn(batch_size, nheads, dtype=dtype, device=device)
        dt_bias = torch.randn(nheads, dtype=dtype, device=device)
        state_indices_tensor = torch.arange(batch_size, dtype=torch.int32, device=device)
        
        A = A[:, None, ...][:, :, None].expand(
            -1, headdim, dstate).to(dtype=torch.float32)
        dt = dt[:, :, None].expand(-1, -1, headdim)
        dt_bias = dt_bias[:, None, ...].expand(-1, headdim)
        D = D[:, None, ...].expand(-1, headdim)
        B = B.view(-1, n_groups, B.shape[1] // n_groups)
        C = C.view(-1, n_groups, C.shape[1] // n_groups)

        initial_states = (
            torch.randn(batch_size, nheads, headdim, dstate, dtype=dtype, device=device)
            if has_initial_state else None
        )

        # sequence_idx = torch.cat(
        #     [ 
        #         torch.zeros((seq_len-batch_size+1,), dtype=torch.int32, device=device),
        #         torch.arange(1, batch_size, dtype=torch.int32, device=device),
        #     ]
        # )
        # print(f"{sequence_idx.shape=}")
        
        # cu_seqlens = torch.cat(
        #     [ 
        #         torch.zeros((1, ), dtype=torch.int32, device=device),
        #         torch.arange(seq_len-batch_size+1, seq_len+1, dtype=torch.int32, device=device),
        #     ]
        # )
        
        # chunk_indices, chunk_offsets = query_start_loc_to_chunk_indices_offsets(
        #     query_start_loc=cu_seqlens, chunk_size=chunk_size, total_seqlens=seq_len)

        return (
            # seq_len,
            hidden_states, initial_states, 
            A, B, C, D, dt, dt_bias,
            state_indices_tensor,
            # cu_seqlens, sequence_idx, chunk_indices, chunk_offsets,
        )

    (
        # seq_len, 
        hidden_states, initial_states, 
        A, B, C, D, dt, dt_bias, 
        state_indices_tensor,
        # cu_seqlens, sequence_idx, chunk_indices, chunk_offsets,
    ) = generate_dummy_data(batch_size)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    # warm up
    warmup_start = time()
    for _ in range(3):
            _ = selective_state_update(
                initial_states,
                hidden_states,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
                state_batch_indices=state_indices_tensor,
            )
    print (f"warmup time {time()-warmup_start:.3f} seconds")
    

    start_time = time()

    if profile_on:
        with profile(activities=activities,with_stack=True) as prof:
            for _ in range(repeats):
                _ = selective_state_update(
                    initial_states,
                    hidden_states,
                    dt,
                    A,
                    B,
                    C,
                    D,
                    z=None,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                    state_batch_indices=state_indices_tensor,
                )

        filename = f"traces/mamba2_decode_trace_b{batch_size}.json"
        prof.export_chrome_trace(filename)
    else:
        
        for _ in range(repeats):
            _ = selective_state_update(
                initial_states,
                hidden_states,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
                state_batch_indices=state_indices_tensor,
            )
        
    torch.cuda.synchronize()
    end_time = time()
    elapsed_time = (end_time - start_time) * 1000 # ms
    print (f"total {elapsed_time=:.3f} mseconds")
    iter_time = elapsed_time / repeats # ms
    print (f"per iter:{iter_time=:.3f} mseconds")

if __name__ == '__main__':
    import fire
    fire.Fire(main)
