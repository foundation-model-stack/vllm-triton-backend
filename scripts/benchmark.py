#  /*******************************************************************************
#   * Copyright 2025 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#


import sys
import os
from typing import List, Optional, Tuple, Union

import pytest
import torch
import triton
import pandas as pd
import numpy as np
import math
import random
from datetime import datetime
from enum import Enum
import itertools
import triton.profiler as proton

from vllm_utils import (
    create_kv_caches_with_random,
    ref_single_query_cached_kv_attention,
    ref_multi_query_kv_attention,
    ref_prefix_prefill,
    ref_reshape_and_cache_flash,
    ref_reshape_and_cache,
    ref_paged_attn,
)
from torch_utils import get_gpu_label, end2end_bench
from ibm_triton_lib.utils.triton_utils import get_runtime_label
from roofline.proton_viewer import parse

# let the three most used variables be overwritten separately
STORE_TEST_RESULT_PATH = os.environ.get("STORE_TEST_RESULT_PATH", None)
MY_IUT = [
    e for e in os.environ.get("MY_IUT", "").split(",") if len(e) > 0
]  # my implementations under test (IUT)
MY_METHODS = [e for e in os.environ.get("MY_METHODS", "").split(",") if len(e) > 0]


class Implementation(Enum):
    VLLM_CUDA_V1 = 0
    VLLM_CUDA_V2 = 1
    TRITON_2D = 2
    BASELINE_TRITON = 3
    XFORMERS = 4
    FLASH_ATTN = 5
    FLASHINFER = 6
    TRITON_FP8 = 7
    TRITON_3D = 8
    TRITON_FUSED = 9
    UNF_TRITON_3D = 10
    UNF_TRITON_2D = 11
    UNF_TRITON_AUTO = 12
    PYTORCH_NATIVE = 13
    TRITON_TUNED = 14
    TRITON_FALLBACK = 15
    UNF_TRITON_2D_SIMPLE = 16
    NT_UNF_TRITON_3D = 17
    NT_UNF_TRITON_2D = 18
    NT_UNF_TRITON_AUTO = 19


class BenchmarkMode(Enum):
    CUDA_EVENTS = 0
    END2END = 1
    CUDA_GRAPHS = 2
    TORCH_COMPILE = 3


class BatchComposition(Enum):
    DEC_PRE = 0
    PRE_DEC = 1
    ALTERNATING = 2


impl_translate = {i.name: i.value for i in Implementation}
method_translate = {i.name: i.value for i in BenchmarkMode}
batch_comp_translate = {i.name: i.value for i in BatchComposition}

# DTYPES = [torch.half, torch.bfloat16, torch.float]
DTYPES = [torch.float16]
SEEDS = [0]

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
# order:  num_query_heads, num_kv_heads
NUM_HEADS = [(32, 32), (32, 8)]
SEQUENCE_LENGTHS = [16, 32, 64, 128, 512, 1024, 2048, 4096]
PREFIX_PREFILL_SHARE_OF_DECODE = [0.0, 0.5, 1.0]
PREFIX_PREFILL_SHARE_OF_PARTIAL_PREFILL = [0.0, 0.5]
PREFIX_PREFILL_BATCH_COMPOSITION = [BatchComposition.ALTERNATING]

HEAD_SIZES = [128]  # only powers of 2! for llama2 & 3
# head_size * head_numbers = hidden_size

BLOCK_SIZES = [16]
# if torch.version.hip:
#     BLOCK_SIZES = [16, 128]
NUM_BLOCKS = [4321]  # "arbitrary values for testing..."

# options most likely not used...but keep for now?
CAUSAL_FLASH = [True]  # vLLM only needs causal=True

PROMPT_PATTERNS = []
PROMPT_PATTERNS.append([1.0])
PROMPT_PATTERNS.append([0.1, 0.4, 0.5, 1.0, 0.2])

STATE_DIM = [128]
STATE_N_GROUPS = [1]
HAS_INITIAL_STATE = [True]

MOE_NUM_EXPERTS = [8]
MOE_N = [14336]  # intermediate size of mixtral-8x7b
MOE_K = [4096]  # for mixtral-8x7b
TP_FACTOR = [1, 2]
MOE_TOP_K = [2]  # for mixtral-8x7b, mixtral-8x22b


IMPLEMENTATION_UT = [
    Implementation.TRITON_2D,
    Implementation.TRITON_3D,
    Implementation.BASELINE_TRITON,
    Implementation.VLLM_CUDA_V1,
    Implementation.VLLM_CUDA_V2,
    # Implementation.XFORMERS,
    Implementation.FLASH_ATTN,
    # Implementation.TRITON_FP8,
    # Implementation.FLASHINFER,
    Implementation.UNF_TRITON_3D,
    Implementation.UNF_TRITON_2D,
    Implementation.UNF_TRITON_AUTO,
]
# MAX_VALUES = [0.01, 0.1, 1.0]
MAX_VALUES = [1.0]
BENCHMARK_MODES = [BenchmarkMode.CUDA_EVENTS, BenchmarkMode.CUDA_GRAPHS]

device = "cuda:0"
gpu_name = get_gpu_label()

do_benchmarks = True
# do_benchmarks = False
quantiles = [0.5, 0.2, 0.8]
# should maybe also be controlled via env variable
force_dump_dataframes = False
enforce_numerical_correctness = True
# enforce_numerical_correctness = False
do_profiling = False  # will add overhead to kernel runtime measured via CUDA_EVENTS
store_hatchet = False
add_triton_dejavu_envs = True
debug_flag = False


test_setup_vars = [
    "SEEDS",
    "BATCH_SIZES",
    "NUM_HEADS",
    "SEQUENCE_LENGTHS",
    "PREFIX_PREFILL_SHARE_OF_DECODE",
    "PREFIX_PREFILL_SHARE_OF_PARTIAL_PREFILL",  # "PREFIX_PREFILL_BATCH_COMPOSITION",
    "HEAD_SIZES",
    "BLOCK_SIZES",
    "NUM_BLOCKS",
    "CAUSAL_FLASH",
    "PROMPT_PATTERNS",
    "MAX_VALUES",
    "STATE_DIM",
    "STATE_N_GROUPS",
    "HAS_INITIAL_STATE",
    "MOE_NUM_EXPERTS",
    "MOE_N",
    "MOE_K",
    "TP_FACTOR",
    "MOE_TOP_K",
]
# "BENCHMARK_MODES", "IMPLEMENTATION_UT" ]
debug_env_vars = [
    "STORE_TEST_RESULT_PATH",
    "TEST_ALLOW_INCORRECT",
    "TRITON_BACKEND_DEBUG",
]

# need to deal with envfile here
if len(sys.argv) >= 1:
    envfile_name = None
    for ca in sys.argv[1:]:
        if ".conf" in ca:
            envfile_name = ca
            break
    if envfile_name is not None:
        from dotenv import dotenv_values
        import json

        envfile_path = os.path.abspath(envfile_name)
        if not os.path.isfile(envfile_path):
            raise RuntimeError(f"Test config file {envfile_path} does not exist.")
        env_setting = dotenv_values(envfile_path)
        if len(env_setting) == 0:
            raise RuntimeError(f"Test config file {envfile_path} does not contain valid configs.")
        print(f"\nApplied test config: {envfile_path}")
        # filter allowed, convert all to lists
        env_setting_filtered = {
            k: json.loads(env_setting[k]) for k in test_setup_vars if k in env_setting
        }
        # update all
        globals().update(env_setting_filtered)
        # fix enums
        if "DTYPES" in env_setting:
            sl = json.loads(env_setting["DTYPES"])
            DTYPES = [getattr(torch, v) for v in sl]
        if "PREFIX_PREFILL_BATCH_COMPOSITION" in env_setting:
            sl = json.loads(env_setting["PREFIX_PREFILL_BATCH_COMPOSITION"])
            PREFIX_PREFILL_BATCH_COMPOSITION = [
                BatchComposition(batch_comp_translate[v]) for v in sl
            ]
        # iut and methods could come here too, or are overwritten below
        if "IMPLEMENTATION_UT" in env_setting:
            sl = json.loads(env_setting["IMPLEMENTATION_UT"])
            IMPLEMENTATION_UT = [Implementation(impl_translate[v]) for v in sl]
        if "BENCHMARK_MODES" in env_setting:
            sl = json.loads(env_setting["BENCHMARK_MODES"])
            BENCHMARK_MODES = [BenchmarkMode(method_translate[v]) for v in sl]
        if "HAS_INITIAL_STATE" in env_setting:
            sl = json.loads(env_setting["HAS_INITIAL_STATE"])
            HAS_INITIAL_STATE = [True if v == "True" else False for v in sl]

        # set additional flags
        if "STORE_TEST_RESULT_PATH" in env_setting and STORE_TEST_RESULT_PATH is None:
            STORE_TEST_RESULT_PATH = env_setting["STORE_TEST_RESULT_PATH"]
        if (
            "TEST_ALLOW_INCORRECT" in env_setting
            and env_setting["TEST_ALLOW_INCORRECT"] == "1"
        ):
            enforce_numerical_correctness = False
        if (
            "TRITON_BACKEND_DEBUG" in env_setting
            and env_setting["TRITON_BACKEND_DEBUG"] == "1"
        ):
            debug_flag = True

if len(MY_IUT) > 0:
    IMPLEMENTATION_UT = []
    for ci_value in MY_IUT:
        IMPLEMENTATION_UT.append(Implementation(impl_translate[ci_value]))
if len(MY_METHODS) > 0:
    BENCHMARK_MODES = []
    for cb_value in MY_METHODS:
        BENCHMARK_MODES.append(BenchmarkMode(method_translate[cb_value]))
# only overwrite the .conf file if the environment variable is present!
if "TEST_ALLOW_INCORRECT" in os.environ:
    enforce_numerical_correctness = os.environ["TEST_ALLOW_INCORRECT"] == "1"
if "TRITON_BACKEND_DEBUG" in os.environ:
    debug_flag = os.environ["TRITON_BACKEND_DEBUG"] == "1"


for varlen_p in PROMPT_PATTERNS:
    for e in varlen_p:
        assert e <= 1.0


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("seqlen", SEQUENCE_LENGTHS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("prompt_pattern", PROMPT_PATTERNS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("implementation", IMPLEMENTATION_UT)
@pytest.mark.parametrize("max_value", MAX_VALUES)
@pytest.mark.parametrize("benchmark_mode", BENCHMARK_MODES)
@torch.inference_mode()
def test_decode_vllm_v0_attention(
    capsys,
    request,
    batch_size,
    num_heads,
    seqlen,
    head_size,
    block_size,
    num_blocks,
    prompt_pattern,
    dtype,
    seed,
    implementation,
    max_value,
    benchmark_mode,
    overwrite_df=None,
    df_file_prefix=None,
    torch_profiling=False,
    prof_filename=None,
):

    my_name = "test_decode_attention"
    my_instance = "test_decode_attention"
    if request is not None:
        my_id = request.node.nodeid.split("::")[-1]
        my_name = my_id.split("[")[0]
        my_instance = my_id.split("[")[1][:-1]
    realistic_prompt_mode = len(prompt_pattern) > 1
    gqa_mode = num_heads[0] != num_heads[1]

    if implementation not in [
        Implementation.BASELINE_TRITON,
        Implementation.FLASH_ATTN,
        Implementation.VLLM_CUDA_V1,
        Implementation.VLLM_CUDA_V2,
        Implementation.TRITON_2D,
        Implementation.TRITON_3D,
        Implementation.TRITON_FP8,
        Implementation.XFORMERS,
        Implementation.FLASHINFER,
    ]:
        pytest.skip("unsupported configuration")

    if implementation == Implementation.BASELINE_TRITON and (
        benchmark_mode == BenchmarkMode.CUDA_GRAPHS or realistic_prompt_mode or gqa_mode
    ):
        pytest.skip("unsupported configuration")

    if implementation == Implementation.TRITON_FP8 and (
        seqlen % block_size != 0 or seqlen == 16
    ):
        pytest.skip("unsupported configuration")
    if implementation == Implementation.XFORMERS and gqa_mode:
        pytest.skip()

    # TODO
    if implementation in [
        Implementation.UNF_TRITON_3D,
        Implementation.UNF_TRITON_2D,
        Implementation.UNF_TRITON_AUTO,
    ]:
        pytest.skip()

    RTOL = 0
    ATOL = min(3.1e-3 * max_value, 1e-3)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    tdev = torch.device(device)
    torch.cuda.set_device(tdev)
    torch.set_default_device(tdev)

    seqlen_fraction = itertools.cycle(prompt_pattern)
    seq_lens = [int(np.ceil(seqlen * next(seqlen_fraction))) for _ in range(batch_size)]
    # NOTE(ngl): Some/all implementations (VLLM_CUDA_V1, XFORMERS, some triton version) assume
    #   there is at least one page per request. That's why apparently the numerical error is
    #   higher at random places if the request is very very small.
    for seq_len in seq_lens:
        if seq_len < block_size:
            ATOL = min(6.2e-3 * max_value, 1e-3)
            break

    kv_cache_dtype = "auto"
    scale = float(1.0 / (head_size**0.5))  # as done by vLLM
    num_query_heads, num_kv_heads = num_heads
    use_alignment_optimization = False
    if implementation in [Implementation.VLLM_CUDA_V1, Implementation.VLLM_CUDA_V2]:
        use_alignment_optimization = True

    # to avoid 'local variable referenced before assignment' when trying to del them
    query = None
    block_tables_lst: List[List[int]] = []
    key_caches = None
    value_caches = None
    key_cache = None
    value_cache = None
    ref_output = None
    output = None
    captured = ""

    inner_exception = None
    try:
        query = torch.empty(batch_size, num_query_heads, head_size, dtype=dtype)
        query.uniform_(-max_value, max_value)

        assert num_query_heads % num_kv_heads == 0
        num_queries_per_kv = num_query_heads // num_kv_heads
        alibi_slopes = None

        max_seq_len = max(seq_lens)
        seq_lens = torch.tensor(seq_lens, dtype=torch.int)

        # Create the block tables.
        max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        if num_blocks < batch_size * max_num_blocks_per_seq:
            pytest.skip("unsupported configuration")

        assert num_blocks >= batch_size * max_num_blocks_per_seq

        for _ in range(batch_size):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
            ]
            block_tables_lst.append(block_table)

        block_tables = torch.tensor(block_tables_lst, dtype=torch.int)

        # Create the KV caches.
        key_caches, value_caches = create_kv_caches_with_random(
            num_blocks,
            block_size,
            1,
            num_kv_heads,
            head_size,
            max_value,
            kv_cache_dtype,
            dtype,
            seed,
            device,
            alignment_optimization=use_alignment_optimization,
        )

        key_cache, value_cache = key_caches[0], value_caches[0]

        ref_output = torch.empty_like(query)
        if not torch_profiling:
            ref_single_query_cached_kv_attention(
                ref_output,
                query,
                num_queries_per_kv,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                scale,
                alibi_slopes,
            )

        if implementation == Implementation.BASELINE_TRITON:
            from callers import BaselineTritonCaller as Caller
        elif implementation == Implementation.TRITON_2D:
            from callers import Triton2dAttentionDecodeCaller as Caller
            from triton_dejavu import global_cache_lock

            global_cache_lock.unlock()
        elif implementation == Implementation.TRITON_3D:
            from callers import Triton3dAttentionDecodeCaller as Caller
        elif implementation == Implementation.TRITON_FP8:
            from callers import TritonFp8Caller as Caller
        elif implementation == Implementation.VLLM_CUDA_V1:
            from callers import VllmCudaV1Caller as Caller
        elif implementation == Implementation.VLLM_CUDA_V2:
            from callers import VllmCudaV2Caller as Caller
        elif implementation == Implementation.XFORMERS:
            from callers import XformersCaller as Caller
        elif implementation == Implementation.FLASH_ATTN:
            from callers import FlashAttnDecodeCaller as Caller
        elif implementation == Implementation.FLASHINFER:
            from callers import FlashInferCaller as Caller
        elif implementation == Implementation.TRITON_FUSED:
            from callers import FusedTritonDecodeOnlyCaller as Caller

        if Caller.requires_allocated_output:
            output = torch.empty_like(query)

        call_func_under_test = Caller.make_call_func(
            output,
            query,
            key_cache,
            value_cache,
            batch_size,
            seq_lens,
            max_seq_len,
            scale,
            block_tables,
            alibi_slopes,
            kv_cache_dtype,
        )

        output_ = call_func_under_test()
        output = Caller.select_output(output, output_)

        if capsys is not None:
            captured_raw = capsys.readouterr()  # returns stdout, stderr
            for l in captured_raw:
                if len(l) > 0:
                    # captured += l  # + '|'
                    captured += l + " "

        # compare
        if not torch_profiling:
            if enforce_numerical_correctness:
                # for better reports
                triton.testing.assert_close(ref_output, output, atol=ATOL, rtol=RTOL)
                allclose_pass = True
            else:
                allclose_pass = torch.allclose(ref_output, output, atol=ATOL, rtol=RTOL)
        else:
            allclose_pass = None

        # benchmark only correct results
        if do_benchmarks:
            if implementation == Implementation.TRITON_2D:
                # switch off compilation...hopefully
                global_cache_lock.lock()
            if overwrite_df is None and my_name not in pytest.global_pds:
                pytest.global_pds[my_name] = pd.DataFrame()
            elif overwrite_df is not None and my_name not in overwrite_df:
                overwrite_df[my_name] = pd.DataFrame()

            profiling_started = False
            if (
                do_profiling
                and implementation
                in [
                    Implementation.TRITON_2D,
                    Implementation.TRITON_3D,
                    Implementation.BASELINE_TRITON,
                ]
                and benchmark_mode == BenchmarkMode.CUDA_EVENTS
                and not torch_profiling
            ):
                if store_hatchet:
                    hatchet_name = os.path.abspath(
                        f"{pytest.global_pd_file_prefix}/{my_name}_profile_{my_instance}"
                    )
                else:
                    hatchet_name = os.path.abspath(
                        f"/tmp/{my_name}_profile_{my_instance}"
                    )
                proton.start(hatchet_name, hook="triton")
                profiling_started = True

            warmup_rep = 25
            bench_rep = 100
            if torch_profiling:
                warmup_rep = 1
                bench_rep = 5
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    with_stack=True,
                ) as prof:
                    torch.cuda.synchronize()
                    ms, min_ms, max_ms = measure_benchmarks(
                        benchmark_mode, call_func_under_test, warmup_rep, bench_rep
                    )
                    torch.cuda.synchronize()
                prof.export_chrome_trace(prof_filename)
            else:
                ms, min_ms, max_ms = measure_benchmarks(
                    benchmark_mode, call_func_under_test, warmup_rep, bench_rep
                )

            proton_count = None
            proton_ns = None
            proton_util_compute = None
            proton_util_bw = None
            if profiling_started:
                proton.finalize()
                # readout
                metrics = ["util_flops", "util_bytes"]
                filter_for = ".*triton.*"
                proton_graph = parse(
                    metrics,
                    f"{hatchet_name}.hatchet",
                    include=filter_for,
                    return_only_df=True,
                )
                proton_df = proton_graph.dataframe
                assert (
                    len(proton_df) == 2
                )  # TODO: update if multiple kernels are called
                proton_count = proton_df.iloc[-1]["count"]
                proton_ns = proton_df.iloc[-1]["time (ns)"]
                proton_util_compute = proton_df.iloc[-1]["util_flops (inc)"]
                proton_util_bw = proton_df.iloc[-1]["util_bytes (inc)"]

            record = {
                "batch_size": batch_size,
                "num_query_heads": num_query_heads,
                "num_kv_heads": num_kv_heads,
                "seqlen": seqlen,
                "head_size": head_size,
                "block_size": block_size,
                "num_blocks": num_blocks,
                "dtype": dtype,
                "max_value": max_value,
                "realistic_prompt_mode": realistic_prompt_mode,
                "gqa_mode": gqa_mode,
                "prompt_pattern": prompt_pattern,
                "implementation": implementation,
                "ms": ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "benchmark_mode": benchmark_mode,
                "allclose_pass": allclose_pass,
                "ATOL": ATOL,
                "RTOL": RTOL,
                "proton_count": proton_count,
                "proton_ns": proton_ns,
                "proton_util_compute": proton_util_compute,
                "proton_util_bw": proton_util_bw,
                "captured": captured,
            }

            if add_triton_dejavu_envs:
                dejavu_envs = {}
                _skip_dejavu_envs = [
                    "_TRITON_DEJAVU_DETERMINED_CUDA_VERSION",
                    "DEBUG",
                    "STORAGE",
                ]
                for env in os.environ.keys():
                    if "TRITON_DEJAVU_" in env:
                        if any([skip_s in env for skip_s in _skip_dejavu_envs]):
                            continue
                        dejavu_envs[env] = os.environ[env]
                record.update(dejavu_envs)

            if torch.version.hip and implementation == Implementation.FLASH_ATTN:
                record["implementation"] = "Implementation.ROCM_FLASH_ATTN"

            if overwrite_df is None:
                pytest.global_pds[my_name] = pd.concat(
                    [pytest.global_pds[my_name], pd.Series(record).to_frame().T]
                ).reset_index(drop=True)

                if pytest.global_pd_file_prefix is not None:
                    filename = os.path.abspath(
                        f"{pytest.global_pd_file_prefix}/{my_name}.csv"
                    )
                    write_df_and_chmod(pytest.global_pds[my_name], filename)
            else:
                overwrite_df[my_name] = pd.concat(
                    [overwrite_df[my_name], pd.Series(record).to_frame().T]
                ).reset_index(drop=True)

                filename = os.path.abspath(f"{df_file_prefix}/{my_name}.csv")
                write_df_and_chmod(overwrite_df[my_name], filename)

    except Exception as e:
        print("\ncaptured:")
        print(captured)
        print("\nexception:")
        print(e)
        inner_exception = e
    finally:
        # cleanup memory
        try:
            del query
            del seq_lens
            del block_tables_lst
            del key_caches
            del value_caches
            del key_cache
            del value_cache
            del ref_output
            del output
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            print(e)
            # pass
        finally:
            if inner_exception is not None:
                raise inner_exception


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("seqlen", SEQUENCE_LENGTHS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("causal", CAUSAL_FLASH)
@pytest.mark.parametrize("prompt_pattern", PROMPT_PATTERNS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("implementation", IMPLEMENTATION_UT)
@pytest.mark.parametrize("max_value", MAX_VALUES)
@pytest.mark.parametrize("benchmark_mode", BENCHMARK_MODES)
@torch.inference_mode()
def test_prefill_vllm_v0_attention(
    capsys,
    request,
    batch_size,
    num_heads,
    seqlen,
    head_size,
    causal,
    prompt_pattern,
    dtype,
    seed,
    implementation,
    max_value,
    benchmark_mode,
):
    my_id = request.node.nodeid.split("::")[-1]
    my_name = my_id.split("[")[0]
    my_instance = my_id.split("[")[1][:-1]
    realistic_prompt_mode = len(prompt_pattern) > 1
    dejavu_tag = os.environ["TRITON_DEJAVU_TAG"]
    fallback_mode = os.getenv("NGL_EXP_FALLBACK", "none")
    gqa_mode = num_heads[0] != num_heads[1]

    if torch.cuda.get_device_capability()[0] < 8:
        # reduce operations are not supported (?)
        pytest.skip()

    # TODO
    if implementation not in [
        Implementation.TRITON_3D,
        Implementation.FLASH_ATTN,
        Implementation.PYTORCH_NATIVE,
    ]:
        pytest.skip("unsupported configuration")
    elif implementation == Implementation.TRITON_3D:
        if (not math.log(head_size, 2).is_integer()) or (head_size > 256):
            pytest.skip()
        if seqlen > 4096 and batch_size > 64:
            # FIXME(ngl): causes RuntimeError: CUDA error: an illegal memory access was encountered
            #  (with triton 3.2.0)
            # for now, we support only batch size of 64 above prompt length of 4096
            pytest.skip()
        if batch_size > 200:
            # FIXME(ngl): also causes illegal memory access
            pytest.skip()
    elif implementation == Implementation.PYTORCH_NATIVE and realistic_prompt_mode:
        pytest.skip("unsupported configuration")

    # TODO
    if implementation in [
        Implementation.UNF_TRITON_3D,
        Implementation.UNF_TRITON_2D,
        Implementation.UNF_TRITON_AUTO,
    ]:
        pytest.skip()

    ATOL = 1e-3 * max_value
    RTOL = 1e-5

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    tdev = torch.device(device)
    torch.cuda.set_device(tdev)
    torch.set_default_device(tdev)

    seqlen_fraction = itertools.cycle(prompt_pattern)
    max_seqlen = seqlen
    seq_lens = [
        int(np.ceil(max_seqlen * next(seqlen_fraction))) for _ in range(batch_size)
    ]
    total_token_num = np.sum(seq_lens)
    kv_cache_dtype = "auto"
    scale = float(1.0 / (head_size**0.5))  # as done by vLLM
    num_query_heads, num_kv_heads = num_heads

    # to avoid 'local variable referenced before assignment' when trying to del them
    query = None
    key_cache = None
    value_cache = None
    M = None
    p = None
    output = None
    ref_output = None
    cu_seqlens_q = None
    seq_start_loc = None
    cu_seqlens_k = None
    captured = ""

    inner_exception = None
    try:
        query = torch.empty(total_token_num, num_query_heads, head_size, dtype=dtype)
        query.uniform_(-max_value, max_value)
        key_cache = torch.empty(total_token_num, num_kv_heads, head_size, dtype=dtype)
        key_cache.uniform_(-max_value, max_value)
        value_cache = torch.empty(total_token_num, num_kv_heads, head_size, dtype=dtype)
        value_cache.uniform_(-max_value, max_value)

        seq_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=tdev)
        prompt_lens_tensor = torch.tensor(seq_lens, dtype=torch.long, device=tdev)
        torch.cumsum(
            prompt_lens_tensor, dim=0, dtype=seq_start_loc.dtype, out=seq_start_loc[1:]
        )

        # reference implementation
        ref_output = ref_multi_query_kv_attention(
            seq_start_loc,
            query,
            key_cache,
            value_cache,
            scale,
            dtype,
            num_kv_heads,
            num_query_heads,
        )

        if implementation == Implementation.FLASH_ATTN:
            from callers import FlashAttnPrefillCaller as Caller
        elif implementation == Implementation.TRITON_3D:
            from callers import Triton3dAttentionPrefillCaller as Caller
        elif implementation == Implementation.PYTORCH_NATIVE:
            from callers import PytorchNativeAttentionPrefillCaller as Caller

        if Caller.requires_allocated_output:
            output = torch.empty_like(query)
        # output is declared already above

        call_func_under_test = Caller.make_call_func(
            output,
            query,
            key_cache,
            value_cache,
            seq_start_loc,
            seq_start_loc,
            max_seqlen,
            max_seqlen,
            scale,
            causal,
        )

        output_ = call_func_under_test()
        output = Caller.select_output(output, output_)

        if capsys is not None:
            captured_raw = capsys.readouterr()  # returns stdout, stderr
            for l in captured_raw:
                if len(l) > 0:
                    # captured += l  # + '|'
                    captured += l + " "

        # compare
        if enforce_numerical_correctness:
            # assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
            # for better reports
            triton.testing.assert_close(ref_output, output, atol=ATOL, rtol=RTOL)
            allclose_pass = True
        else:
            allclose_pass = torch.allclose(ref_output, output, atol=ATOL, rtol=RTOL)

        # benchmark only correct results
        if do_benchmarks:
            if my_name not in pytest.global_pds:
                pytest.global_pds[my_name] = pd.DataFrame()

            profiling_started = False
            if (
                do_profiling
                and implementation in [Implementation.TRITON_2D]  # TODO
                and benchmark_mode == BenchmarkMode.CUDA_EVENTS
            ):
                if store_hatchet:
                    hatchet_name = os.path.abspath(
                        f"{pytest.global_pd_file_prefix}/{my_name}_profile_{my_instance}"
                    )
                else:
                    hatchet_name = os.path.abspath(
                        f"/tmp/{my_name}_profile_{my_instance}"
                    )
                proton.start(hatchet_name, hook="triton")
                profiling_started = True

            # equals to defaults
            warmup_rep = 25
            bench_rep = 100
            ms, min_ms, max_ms = measure_benchmarks(
                benchmark_mode, call_func_under_test, warmup_rep, bench_rep
            )

            proton_count = None
            proton_ns = None
            proton_util_compute = None
            proton_util_bw = None
            if profiling_started:
                proton.finalize()
                # readout
                metrics = ["util_flops", "util_bytes"]
                filter_for = ".*triton.*"
                proton_graph = parse(
                    metrics,
                    f"{hatchet_name}.hatchet",
                    include=filter_for,
                    return_only_df=True,
                )
                proton_df = proton_graph.dataframe
                assert (
                    len(proton_df) == 2
                )  # TODO: update if multiple kernels are called
                proton_count = proton_df.iloc[-1]["count"]
                proton_ns = proton_df.iloc[-1]["time (ns)"]
                proton_util_compute = proton_df.iloc[-1]["util_flops (inc)"]
                proton_util_bw = proton_df.iloc[-1]["util_bytes (inc)"]

            record = {
                "batch_size": batch_size,
                "num_query_heads": num_query_heads,
                "num_kv_heads": num_kv_heads,
                "seqlen": max_seqlen,
                "head_size": head_size,
                "dtype": dtype,
                "causal": causal,
                "max_value": max_value,
                "realistic_prompt_mode": realistic_prompt_mode,
                "gqa_mode": gqa_mode,
                "prompt_pattern": prompt_pattern,
                "implementation": implementation,
                "dejavu_tag": dejavu_tag,
                "dejavu_fallback_mode": fallback_mode,
                "ms": ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "benchmark_mode": benchmark_mode,
                "allclose_pass": allclose_pass,
                "ATOL": ATOL,
                "RTOL": RTOL,
                "proton_count": proton_count,
                "proton_ns": proton_ns,
                "proton_util_compute": proton_util_compute,
                "proton_util_bw": proton_util_bw,
                "captured": captured,
            }

            if add_triton_dejavu_envs:
                dejavu_envs = {}
                _skip_dejavu_envs = [
                    "_TRITON_DEJAVU_DETERMINED_CUDA_VERSION",
                    "DEBUG",
                    "STORAGE",
                ]
                for env in os.environ.keys():
                    if "TRITON_DEJAVU_" in env:
                        if any([skip_s in env for skip_s in _skip_dejavu_envs]):
                            continue
                        dejavu_envs[env] = os.environ[env]
                record.update(dejavu_envs)

            if torch.version.hip and implementation == Implementation.FLASH_ATTN:
                record["implementation"] = "Implementation.ROCM_FLASH_ATTN"

            pytest.global_pds[my_name] = pd.concat(
                [pytest.global_pds[my_name], pd.Series(record).to_frame().T]
            ).reset_index(drop=True)

            if pytest.global_pd_file_prefix is not None:
                filename = os.path.abspath(
                    f"{pytest.global_pd_file_prefix}/{my_name}.csv"
                )
                write_df_and_chmod(pytest.global_pds[my_name], filename)

    except Exception as e:
        print("\ncaptured:")
        print(captured)
        print("\nexception:")
        print(e)
        inner_exception = e
    finally:
        # cleanup memory
        try:
            del query
            del key_cache
            del value_cache
            del p
            del M
            del output
            del ref_output
            del seq_lens
            del prompt_lens_tensor
            del cu_seqlens_q
            del seq_start_loc
            del cu_seqlens_k
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            print(e)
            # pass
        finally:
            if inner_exception is not None:
                raise inner_exception


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
# @pytest.mark.parametrize("querylen", QUERY_LENGTHS)
# @pytest.mark.parametrize("ctxlen", CONTEXT_LENGTHS)
@pytest.mark.parametrize("seqlen", SEQUENCE_LENGTHS)
@pytest.mark.parametrize("decode_share", PREFIX_PREFILL_SHARE_OF_DECODE)
@pytest.mark.parametrize(
    "partial_prefill_share", PREFIX_PREFILL_SHARE_OF_PARTIAL_PREFILL
)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("prompt_pattern", PROMPT_PATTERNS)
@pytest.mark.parametrize("batch_composition", PREFIX_PREFILL_BATCH_COMPOSITION)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("implementation", IMPLEMENTATION_UT)
@pytest.mark.parametrize("max_value", MAX_VALUES)
@pytest.mark.parametrize("benchmark_mode", BENCHMARK_MODES)
@torch.inference_mode()
def test_prefix_vllm_v1_attention(
    capsys,
    request,
    batch_size,
    num_heads,
    # querylen,  # max querylen
    # ctxlen,  # max contextlen
    seqlen,
    decode_share,
    partial_prefill_share,
    head_size,
    block_size,
    num_blocks,
    prompt_pattern,
    batch_composition,
    dtype,
    seed,
    implementation,
    max_value,
    benchmark_mode,
):
    my_id = request.node.nodeid.split("::")[-1]
    my_name = my_id.split("[")[0]
    my_instance = my_id.split("[")[1][:-1]
    realistic_prompt_mode = len(prompt_pattern) > 1
    gqa_mode = num_heads[0] != num_heads[1]

    if torch.cuda.get_device_capability()[0] < 8:
        # reduce operations are not supported (?)
        pytest.skip()

    if implementation not in [
        Implementation.BASELINE_TRITON,
        Implementation.FLASH_ATTN,
        Implementation.TRITON_FUSED,
        Implementation.TRITON_2D,
        Implementation.UNF_TRITON_3D,
        Implementation.UNF_TRITON_2D,
        Implementation.UNF_TRITON_2D_SIMPLE,
        Implementation.UNF_TRITON_AUTO,
        Implementation.NT_UNF_TRITON_3D,
        Implementation.NT_UNF_TRITON_2D,
        Implementation.NT_UNF_TRITON_AUTO,
    ]:
        pytest.skip()

    # TODO: Error: "Offset increment outside graph capture"
    #  for triton and flash_attn
    # if benchmark_mode == BenchmarkMode.CUDA_GRAPHS:
    #     pytest.skip("not supported")

    # TODO
    # RTOL = 0
    # ATOL = min(3.1e-3 * max_value, 1e-3)
    # ATOL = 1e-1 * max_value
    # ATOL = min(2.5 * max_value, 2.5e-2)  # 2.5 for 0.000503% of the values
    ATOL = 0.34 * max_value  # 3.4 for 3.46e-05% of some values
    if realistic_prompt_mode:
        ATOL *= 2.2  # for 0.0313% of the cases...
    RTOL = 1e-5
    # TODO?? due to incomplete output batch?
    if decode_share != 1.0:
        ATOL = 2 * max_value

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    tdev = torch.device(device)
    torch.cuda.set_device(tdev)
    torch.set_default_device(tdev)

    len_fraction = itertools.cycle(prompt_pattern)
    init_seq_lens = [
        int(np.ceil(seqlen * next(len_fraction))) for _ in range(batch_size)
    ]
    decode_seqs = int(np.ceil(batch_size * decode_share))
    prefill_seqs = batch_size - decode_seqs
    partial_prefill_seqs = int(np.ceil(prefill_seqs * partial_prefill_share))
    full_prefill_seqs = prefill_seqs - partial_prefill_seqs

    # reuse same prompt pattern for partial prompts, but with half the length
    len_fraction_half = itertools.cycle([pp * 0.5 for pp in prompt_pattern])
    raw_partial_prefill_ctx_lens = [
        int(np.ceil(l // block_size * next(len_fraction_half))) * block_size
        for l in init_seq_lens
    ]
    partial_prefill_ctx_lens = []
    # avoid "full" partial prefills (i.e. no query left)
    for i in range(batch_size):
        rpl = raw_partial_prefill_ctx_lens[i]
        il = init_seq_lens[i]
        if rpl == il:
            partial_prefill_ctx_lens.append(rpl - block_size)
        else:
            partial_prefill_ctx_lens.append(rpl)
    partial_prefill_q_lens = [
        init_seq_lens[i] - partial_prefill_ctx_lens[i] for i in range(batch_size)
    ]

    query_lens = (
        [1] * decode_seqs
        + partial_prefill_q_lens[decode_seqs : decode_seqs + partial_prefill_seqs]
        + init_seq_lens[decode_seqs + partial_prefill_seqs :]
    )
    ctx_lens = (
        # TODO: subtract one from query length or not? (adapt assert below if changing)
        [ol - 1 for ol in init_seq_lens[:decode_seqs]]
        # init_seq_lens[:decode_seqs]
        + partial_prefill_ctx_lens[decode_seqs : decode_seqs + partial_prefill_seqs]
        + [0] * full_prefill_seqs
    )
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    max_seq_len = max(seq_lens)

    # BatchComposition.DEC_PRE is default
    if batch_composition == BatchComposition.PRE_DEC:
        query_lens.reverse()
        ctx_lens.reverse()
        seq_lens.reverse()
    if batch_composition == BatchComposition.ALTERNATING:
        alorder = []
        indexs_remaining = list(range(len(query_lens)))
        for i in range(len(query_lens) // 2):
            alorder.append(i)
            alorder.append(len(query_lens) - i - 1)
            indexs_remaining.remove(i)
            indexs_remaining.remove(len(query_lens) - i - 1)
        alorder.extend(indexs_remaining)
        query_lens = [query_lens[i] for i in alorder]
        ctx_lens = [ctx_lens[i] for i in alorder]
        seq_lens = [seq_lens[i] for i in alorder]

    if debug_flag:
        print(
            f"decode share: {decode_share}; prefill share {1-decode_share} -> of that: partial prefill share {partial_prefill_share}"
        )
        print(
            f"decode requests: {decode_seqs}; prefill requests: {prefill_seqs} (of that {partial_prefill_seqs} partial prefills and {full_prefill_seqs} full prefills)"
        )
        print("init_seq_lens ", init_seq_lens)
        print("partial_prefill_q_lens ", partial_prefill_q_lens)
        print("partial_prefill_ctx_lens", partial_prefill_ctx_lens)
        print(f"\nAfter assembling the final batch:")
        print(f"\tquery_lens: {query_lens}")
        print(f"\t  ctx_lens: {ctx_lens}")
        print(f"\t  seq_lens: {seq_lens}")
    assert len(ctx_lens) == len(query_lens)
    if not realistic_prompt_mode:
        # assert max_seq_len == seqlen or max_seq_len == seqlen + 1
        assert max_seq_len == seqlen

    # NOTE(ngl): Some/all implementations apparently assume there is at least
    #   one page per decode-request. That's why apparently the numerical error
    #   is higher at random places if the request is very very small.
    for seq_len in seq_lens:
        if seq_len < block_size:
            # ATOL = min(6.2e-3 * max_value, 1e-3)
            # TODO
            ATOL = max(max_value * 1.7, ATOL)
            break

    cache_dtype = dtype  # TODO
    scale = float(1.0 / (head_size**0.5))  # as done by vLLM
    num_query_heads, num_kv_heads = num_heads
    total_token_num = np.sum(seq_lens)
    total_query_tokens = np.sum(query_lens)

    # to avoid 'local variable referenced before assignment' when trying to del them
    query = None
    block_tables_lst: List[List[int]] = []
    block_table_t = None
    b_seq_lens = None
    b_ctx_lens = None
    b_query_lens = None
    b_start_loc = None
    b_seq_start_loc = None
    key_cache = None
    value_cache = None
    key = None
    value = None
    ref_output = None
    output = None
    captured = ""

    inner_exception = None
    try:
        query = torch.empty(total_query_tokens, num_query_heads, head_size, dtype=dtype)
        query.uniform_(-max_value, max_value)

        key = torch.empty(total_token_num, num_kv_heads, head_size, dtype=dtype)
        key.uniform_(-max_value, max_value)
        value = torch.empty(total_token_num, num_kv_heads, head_size, dtype=dtype)
        value.uniform_(-max_value, max_value)

        assert num_query_heads % num_kv_heads == 0
        num_queries_per_kv = num_query_heads // num_kv_heads
        alibi_slopes = None

        b_seq_lens = torch.tensor(
            seq_lens, dtype=torch.int
        )  # vllm unit test uses long, but that doesn't work with flash_attn
        b_ctx_lens = torch.tensor(ctx_lens, dtype=torch.int)
        b_query_lens = torch.tensor(query_lens, dtype=torch.int)
        b_start_loc = torch.cumsum(
            torch.tensor([0] + query_lens, dtype=torch.int), dim=0, dtype=torch.int
        )
        b_seq_start_loc = torch.cumsum(
            torch.tensor([0] + seq_lens, dtype=torch.int), dim=0, dtype=torch.int
        )

        # Create the block tables.
        max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        if num_blocks < batch_size * max_num_blocks_per_seq:
            pytest.skip("unsupported configuration")

        assert num_blocks >= batch_size * max_num_blocks_per_seq

        for _ in range(batch_size):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
            ]
            block_tables_lst.append(block_table)

        block_table_t = torch.tensor(block_tables_lst, dtype=torch.int)

        # Create the KV caches.
        key_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, head_size, dtype=cache_dtype
        )
        value_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, head_size, dtype=cache_dtype
        )
        # reverse engineer slot mapping
        slot_mapping_lst = []
        for i in range(batch_size):
            block_table = block_tables_lst[i]
            slot_mapping_i = []
            for j in range(seq_lens[i]):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size
                slot_number = block_number * block_size + block_offset
                slot_mapping_i.append(slot_number)
            slot_mapping_lst.extend(slot_mapping_i)
        slot_mapping_t = torch.tensor(slot_mapping_lst, dtype=torch.int)

        ref_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping_t,
            block_size,
            total_token_num,
        )

        ref_output = ref_prefix_prefill(
            query,
            num_queries_per_kv,
            key_cache,
            value_cache,
            key,
            value,
            block_table_t,
            b_seq_lens,
            b_ctx_lens,
            b_query_lens,
            b_start_loc,
            batch_size,
            scale,
            dtype,
        )
        # ref_output = ref_paged_attn(
        #     query,
        #     key_cache,
        #     value_cache,
        #     b_query_lens,
        #     b_ctx_lens,
        #     block_table_t,
        #     scale,
        # )

        if implementation == Implementation.FLASH_ATTN:
            from callers import FlashAttnPrefixPrefillCaller as Caller
        elif implementation == Implementation.BASELINE_TRITON:
            from callers import BaselineTritonPrefixPrefillCaller as Caller
        elif implementation == Implementation.TRITON_FUSED:
            from callers import FusedTritonChunkedPrefixPrefill25dCaller as Caller
        elif implementation == Implementation.TRITON_2D:
            from callers import Triton2dChunkedPrefillCaller as Caller
        # elif implementation == Implementation.TRITON_3D:
        #     from callers import Triton3dAttentionDecodeCaller as Caller
        elif implementation == Implementation.UNF_TRITON_3D:
            from callers import UnifiedTriton3dAttentionCaller as Caller
        elif implementation == Implementation.UNF_TRITON_2D:
            from callers import UnifiedTriton2dAttentionCaller as Caller
        elif implementation == Implementation.UNF_TRITON_2D_SIMPLE:
            from callers import SimpleUnifiedTriton2dAttentionCaller as Caller
        elif implementation == Implementation.UNF_TRITON_AUTO:
            from callers import UnifiedTritonAutoAttentionCaller as Caller
        elif implementation == Implementation.NT_UNF_TRITON_3D:
            from callers import NewTilesUnifiedTriton3dAttentionCaller as Caller
        elif implementation == Implementation.NT_UNF_TRITON_2D:
            from callers import NewTilesUnifiedTriton2dAttentionCaller as Caller
        elif implementation == Implementation.NT_UNF_TRITON_AUTO:
            from callers import NewTilesUnifiedTritonAutoAttentionCaller as Caller

        if Caller.requires_allocated_output:
            output = torch.empty_like(query)

        if implementation in [
            Implementation.BASELINE_TRITON,
            Implementation.TRITON_FUSED,
            Implementation.TRITON_2D,
        ]:
            # TODO: better here than in callers? That would mean the caller interfaces are not totally fixed, as with vllm's forwards
            x = 8
            key_cache = torch.zeros(
                num_blocks,
                num_kv_heads,
                head_size // x,
                block_size,
                x,
                dtype=key_cache.dtype,
            )
            value_cache = torch.zeros(
                num_blocks,
                num_kv_heads,
                head_size,
                block_size,
                dtype=value_cache.dtype,
            )
            ref_reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping_t,
                block_size,
                total_token_num,
            )

        call_func_under_test = Caller.make_call_func(
            output,
            query,
            key_cache,
            value_cache,
            key,
            value,
            block_table_t,
            b_seq_lens,
            b_ctx_lens,
            b_query_lens,
            b_start_loc,
            b_seq_start_loc,
            scale,
            # kv_cache_dtype,
        )

        output_ = call_func_under_test()
        output = Caller.select_output(output, output_)

        if capsys is not None:
            captured_raw = capsys.readouterr()  # returns stdout, stderr
            for l in captured_raw:
                if len(l) > 0:
                    # captured += l  # + '|'
                    captured += l + " "
        # compare
        if enforce_numerical_correctness:
            # for better reports
            triton.testing.assert_close(ref_output, output, atol=ATOL, rtol=RTOL)
            allclose_pass = True
        else:
            allclose_pass = torch.allclose(ref_output, output, atol=ATOL, rtol=RTOL)

        # benchmark only correct results
        if do_benchmarks:
            if my_name not in pytest.global_pds:
                pytest.global_pds[my_name] = pd.DataFrame()

            profiling_started = False
            if (
                do_profiling
                and implementation
                in [  # TODO
                    # Implementation.TRITON_2D,
                    # Implementation.TRITON_3D,
                    # Implementation.BASELINE_TRITON,
                ]
                and benchmark_mode == BenchmarkMode.CUDA_EVENTS
            ):
                if store_hatchet:
                    hatchet_name = os.path.abspath(
                        f"{pytest.global_pd_file_prefix}/{my_name}_profile_{my_instance}"
                    )
                else:
                    hatchet_name = os.path.abspath(
                        f"/tmp/{my_name}_profile_{my_instance}"
                    )
                proton.start(hatchet_name, hook="triton")
                profiling_started = True

            # equals to defaults
            warmup_rep = 25
            bench_rep = 100
            ms, min_ms, max_ms = measure_benchmarks(
                benchmark_mode, call_func_under_test, warmup_rep, bench_rep
            )

            proton_count = None
            proton_ns = None
            proton_util_compute = None
            proton_util_bw = None
            if profiling_started:
                proton.finalize()
                # readout
                metrics = ["util_flops", "util_bytes"]
                filter_for = ".*triton.*"
                proton_graph = parse(
                    metrics,
                    f"{hatchet_name}.hatchet",
                    include=filter_for,
                    return_only_df=True,
                )
                proton_df = proton_graph.dataframe
                assert (
                    len(proton_df) == 2
                )  # TODO: update if multiple kernels are called
                proton_count = proton_df.iloc[-1]["count"]
                proton_ns = proton_df.iloc[-1]["time (ns)"]
                proton_util_compute = proton_df.iloc[-1]["util_flops (inc)"]
                proton_util_bw = proton_df.iloc[-1]["util_bytes (inc)"]

            record = {
                "batch_size": batch_size,
                "num_query_heads": num_query_heads,
                "num_kv_heads": num_kv_heads,
                # "querylen": querylen,
                # "ctxlen": ctxlen,
                "seqlen": seqlen,
                "decode_share": decode_share,
                "partial_prefill_share": partial_prefill_share,
                "head_size": head_size,
                "block_size": block_size,
                "num_blocks": num_blocks,
                "dtype": dtype,
                "max_value": max_value,
                "realistic_prompt_mode": realistic_prompt_mode,
                "gqa_mode": gqa_mode,
                "prompt_pattern": prompt_pattern,
                "implementation": implementation,
                "ms": ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "benchmark_mode": benchmark_mode,
                "allclose_pass": allclose_pass,
                "ATOL": ATOL,
                "RTOL": RTOL,
                "proton_count": proton_count,
                "proton_ns": proton_ns,
                "proton_util_compute": proton_util_compute,
                "proton_util_bw": proton_util_bw,
                "captured": captured,
            }

            if add_triton_dejavu_envs:
                dejavu_envs = {}
                _skip_dejavu_envs = [
                    "_TRITON_DEJAVU_DETERMINED_CUDA_VERSION",
                    "DEBUG",
                    "STORAGE",
                ]
                for env in os.environ.keys():
                    if "TRITON_DEJAVU_" in env:
                        if any([skip_s in env for skip_s in _skip_dejavu_envs]):
                            continue
                        dejavu_envs[env] = os.environ[env]
                record.update(dejavu_envs)

            pytest.global_pds[my_name] = pd.concat(
                [pytest.global_pds[my_name], pd.Series(record).to_frame().T]
            ).reset_index(drop=True)

            if pytest.global_pd_file_prefix is not None:
                filename = os.path.abspath(
                    f"{pytest.global_pd_file_prefix}/{my_name}.csv"
                )
                write_df_and_chmod(pytest.global_pds[my_name], filename)

    except Exception as e:
        print("\ncaptured:")
        print(captured)
        print("\nexception:")
        print(e)
        inner_exception = e
    finally:
        # cleanup memory
        try:
            del query
            del b_ctx_lens
            del b_seq_lens
            del b_query_lens
            del b_start_loc
            del b_seq_start_loc
            del block_tables_lst
            del block_table_t
            del key
            del value
            del key_cache
            del value_cache
            del ref_output
            del output
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            print(e)
            # pass
        finally:
            if inner_exception is not None:
                raise inner_exception


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dstate", STATE_DIM)
@pytest.mark.parametrize("n_groups", STATE_N_GROUPS)
@pytest.mark.parametrize("has_initial_state", HAS_INITIAL_STATE)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
# @pytest.mark.parametrize("implementation", IMPLEMENTATION_UT)
@pytest.mark.parametrize("benchmark_mode", BENCHMARK_MODES)
@torch.inference_mode()
def test_mamba_ssm(
    capsys,
    request,
    batch_size,
    num_heads,
    head_size,
    dstate,
    n_groups,
    has_initial_state,
    dtype,
    seed,
    # implementation,
    benchmark_mode,
):
    my_id = request.node.nodeid.split("::")[-1]
    my_name = my_id.split("[")[0]
    my_instance = my_id.split("[")[1][:-1]

    # if torch.cuda.get_device_capability()[0] < 8:
    #     # reduce operations are not supported (?)
    #     pytest.skip()

    # TODO
    assert num_heads[0] == num_heads[1]
    nheads = num_heads[0]
    headdim = head_size

    def generate_dummy_data(batch_size):

        hidden_states = torch.randn(
            batch_size, nheads, headdim, dtype=dtype, device=device
        )
        A = torch.rand(nheads, dtype=dtype, device=device)
        B = torch.randn(batch_size, dstate, dtype=dtype, device=device)
        C = torch.randn(batch_size, dstate, dtype=dtype, device=device)
        D = torch.randn(nheads, dtype=dtype, device=device)
        dt = torch.randn(batch_size, nheads, dtype=dtype, device=device)
        dt_bias = torch.randn(nheads, dtype=dtype, device=device)
        state_indices_tensor = torch.arange(
            batch_size, dtype=torch.int32, device=device
        )

        A = (
            A[:, None, ...][:, :, None]
            .expand(-1, headdim, dstate)
            .to(dtype=torch.float32)
        )
        dt = dt[:, :, None].expand(-1, -1, headdim)
        dt_bias = dt_bias[:, None, ...].expand(-1, headdim)
        D = D[:, None, ...].expand(-1, headdim)
        B = B.view(-1, n_groups, B.shape[1] // n_groups)
        C = C.view(-1, n_groups, C.shape[1] // n_groups)

        initial_states = (
            torch.randn(batch_size, nheads, headdim, dstate, dtype=dtype, device=device)
            if has_initial_state
            else None
        )

        return (
            hidden_states,
            initial_states,
            A,
            B,
            C,
            D,
            dt,
            dt_bias,
            state_indices_tensor,
        )

    captured = ""
    try:
        (
            hidden_states,
            initial_states,
            A,
            B,
            C,
            D,
            dt,
            dt_bias,
            state_indices_tensor,
        ) = generate_dummy_data(batch_size)

        from ibm_triton_lib.kernels import selective_state_update

        # TODO?
        # warm up
        warmup_start = datetime.now()
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
        print(f"warmup time {datetime.now()-warmup_start}")

        if capsys is not None:
            captured_raw = capsys.readouterr()  # returns stdout, stderr
            for l in captured_raw:
                if len(l) > 0:
                    # captured += l  # + '|'
                    captured += l + " "
        # # compare
        # if enforce_numerical_correctness:
        #     # for better reports
        #     triton.testing.assert_close(ref_output, output, atol=ATOL, rtol=RTOL)
        #     allclose_pass = True
        # else:
        #     allclose_pass = torch.allclose(ref_output, output, atol=ATOL, rtol=RTOL)

        call_func_under_test = lambda: selective_state_update(
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

        # benchmark only correct results
        if do_benchmarks:
            if my_name not in pytest.global_pds:
                pytest.global_pds[my_name] = pd.DataFrame()

            # equals to defaults
            warmup_rep = 25
            bench_rep = 100
            ms, min_ms, max_ms = measure_benchmarks(
                benchmark_mode, call_func_under_test, warmup_rep, bench_rep
            )

            record = {
                "batch_size": batch_size,
                "num_heads": nheads,
                "head_size": head_size,
                "dstate": dstate,
                "n_groups": n_groups,
                "has_initial_state": has_initial_state,
                "dtype": dtype,
                # "implementation": implementation,
                "ms": ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "benchmark_mode": benchmark_mode,
                # "allclose_pass": allclose_pass,
                # "ATOL": ATOL,
                # "RTOL": RTOL,
                # "proton_count": proton_count,
                # "proton_ns": proton_ns,
                # "proton_util_compute": proton_util_compute,
                # "proton_util_bw": proton_util_bw,
                "captured": captured,
            }

            if add_triton_dejavu_envs:
                dejavu_envs = {}
                _skip_dejavu_envs = [
                    "_TRITON_DEJAVU_DETERMINED_CUDA_VERSION",
                    "DEBUG",
                    "STORAGE",
                ]
                for env in os.environ.keys():
                    if "TRITON_DEJAVU_" in env:
                        if any([skip_s in env for skip_s in _skip_dejavu_envs]):
                            continue
                        dejavu_envs[env] = os.environ[env]
                record.update(dejavu_envs)

            pytest.global_pds[my_name] = pd.concat(
                [pytest.global_pds[my_name], pd.Series(record).to_frame().T]
            ).reset_index(drop=True)

            if pytest.global_pd_file_prefix is not None:
                filename = os.path.abspath(
                    f"{pytest.global_pd_file_prefix}/{my_name}.csv"
                )
                write_df_and_chmod(pytest.global_pds[my_name], filename)

    except Exception as e:
        print("\ncaptured:")
        print(captured)
        print("\nexception:")
        print(e)
        raise e


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seqlen", SEQUENCE_LENGTHS)
@pytest.mark.parametrize("n", MOE_N)
@pytest.mark.parametrize("k", MOE_K)
@pytest.mark.parametrize("e", MOE_NUM_EXPERTS)
@pytest.mark.parametrize("tp", TP_FACTOR)
@pytest.mark.parametrize("topk", MOE_TOP_K)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("max_value", MAX_VALUES)
@pytest.mark.parametrize("implementation", IMPLEMENTATION_UT)
@pytest.mark.parametrize("benchmark_mode", BENCHMARK_MODES)
def test_fused_moe(
    capsys,
    request,
    batch_size,
    seqlen,
    n: int,
    k: int,
    e: int,
    tp: int,
    topk: int,
    dtype: torch.dtype,
    seed,
    max_value,
    implementation,
    benchmark_mode,
):
    # based on: https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_moe.py
    from vllm.model_executor.layers.activation import SiluAndMul

    my_id = request.node.nodeid.split("::")[-1]
    my_name = my_id.split("[")[0]
    my_instance = my_id.split("[")[1][:-1]

    if implementation not in [Implementation.TRITON_TUNED, Implementation.TRITON_FALLBACK]:
        pytest.skip()
   
    def torch_moe(a, w1, w2, score, topk):
        B, D = a.shape
        a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
        out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)
        topk_weight = topk_weight.view(-1)
        topk_ids = topk_ids.view(-1)
        for i in range(w1.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = SiluAndMul()(
                    a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
        return (out.view(B, -1, w2.shape[1]) *
                topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)

    from ibm_triton_lib.kernels import fused_moe
    # my_experts = TritonExperts()
    # fused_moe = my_experts.apply

    torch.manual_seed(seed)
    tdev = torch.device(device)
    torch.cuda.set_device(tdev)
    # m = batch_size * seqlen
    num_tokens = batch_size * seqlen
    m = num_tokens
    n  = int(n//tp)
    
    # ATOL = 1e-2
    # TODO
    ATOL = max(1e-2, 2 * max_value)
    RTOL = 0

    a = None
    w1 = None
    w2 = None
    score = None
    torch_output = None
    triton_output = None

    inner_exception = None 
    try: 

        a = torch.randn((m, k), device=tdev, dtype=dtype).normal_(mean=0.0, std=0.5 * max_value)
        # w1 = torch.randn((e, n, k), device=tdev, dtype=dtype).normal_(mean=0.0, std=0.5 * max_value)
        # w2 = torch.randn((e, k, n//2), device=tdev, dtype=dtype).normal_(mean=0.0, std=0.5 * max_value)
        w1 = torch.randn((e, 2 * n, k), device=tdev, dtype=dtype).normal_(mean=0.0, std=0.5 * max_value)
        w2 = torch.randn((e, k, n), device=tdev, dtype=dtype).normal_(mean=0.0, std=0.5 * max_value)
        score = torch.randn((m, e), device=tdev, dtype=dtype)
    
        input_gating = torch.empty(num_tokens, e, dtype=torch.float32, device=tdev)

        if enforce_numerical_correctness:
            torch_output = torch_moe(a, w1, w2, score, topk)
            assert torch_output is not None
        """
        from fused_moe.py
         Key Parameters:
            - A: The input tensor representing tokens with shape (*, K), where '*' can
                be any shape representing batches and K is the feature dimension of
                each token.
            - B: The stacked MOE weight tensor with shape (E, N, K), where E is
                the number of experts, K is the input feature dimension, and N is
                the output feature dimension.
            - C: The output cache tensor with shape (M, topk, N), where M is the
                total number of tokens post padding, topk is the number of times
                each token is repeated, and N is the output feature dimension.
        """

        
        use_default_config = True if implementation == Implementation.TRITON_FALLBACK else False
        triton_output = fused_moe(a, w1, w2, input_gating, topk,
                                  renormalize=True, use_default_config=use_default_config) #inplace=True ? 
        assert triton_output is not None
        
        captured = ''
        if capsys is not None:
            captured_raw = capsys.readouterr()  # returns stdout, stderr
            for l in captured_raw:
                if len(l) > 0:
                    # captured += l  # + '|'
                    captured += l  + ' '
        
        # compare
        allclose_pass = float('nan')
        if enforce_numerical_correctness:
            triton.testing.assert_close(torch_output, triton_output, atol=ATOL, rtol=RTOL)
            allclose_pass = True

        call_func_under_test = lambda: fused_moe(a, w1, w2, input_gating, topk, 
                                                 renormalize=True, inplace=True,
                                                 use_default_config=use_default_config)

        # benchmark only correct results
        if do_benchmarks:
            if my_name not in pytest.global_pds:
                pytest.global_pds[my_name] = pd.DataFrame()

            # equals to defaults
            warmup_rep = 25
            bench_rep = 100
            ms, min_ms, max_ms = measure_benchmarks(
                benchmark_mode, call_func_under_test, warmup_rep, bench_rep
            )

            record = {
                "batch_size": batch_size,
                "seqlen": seqlen,
                "num_tokens": num_tokens, # redundant?
                "N": n,
                "K": k,
                "E": e,
                "TP": tp,
                "topk": topk,
                "max_value": max_value,
                "dtype": dtype,
                "implementation": implementation,
                "ms": ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "benchmark_mode": benchmark_mode,
                "allclose_pass": allclose_pass,
                "ATOL": ATOL,
                "RTOL": RTOL,
                # "proton_count": proton_count,
                # "proton_ns": proton_ns,
                # "proton_util_compute": proton_util_compute,
                # "proton_util_bw": proton_util_bw,
                "captured": captured,
            }

            if add_triton_dejavu_envs:
                dejavu_envs = {}
                _skip_dejavu_envs = [
                    "_TRITON_DEJAVU_DETERMINED_CUDA_VERSION",
                    "DEBUG",
                    "STORAGE",
                ]
                for env in os.environ.keys():
                    if "TRITON_DEJAVU_" in env:
                        if any([skip_s in env for skip_s in _skip_dejavu_envs]):
                            continue
                        dejavu_envs[env] = os.environ[env]
                record.update(dejavu_envs)

            pytest.global_pds[my_name] = pd.concat(
                [pytest.global_pds[my_name], pd.Series(record).to_frame().T]
            ).reset_index(drop=True)

            if pytest.global_pd_file_prefix is not None:
                filename = os.path.abspath(
                    f"{pytest.global_pd_file_prefix}/{my_name}.csv"
                )
                write_df_and_chmod(pytest.global_pds[my_name], filename)

    except Exception as e:
        print(e)
        inner_exception = e
    finally:
        # cleanup memory
        try:
            del a
            del w1
            del w2
            del score
            del triton_output
            del torch_output
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            print(e)
            # pass
        finally:
            if inner_exception is not None:
                raise inner_exception




def measure_benchmarks(
    benchmark_mode, call_func_under_test, warmup_rep=25, bench_rep=100
):

    if benchmark_mode == BenchmarkMode.TORCH_COMPILE:
        compiled_fn = torch.compile(call_func_under_test)
        # shortening trace?
        # TODO: necessary? twice?
        compiled_fn()
        compiled_fn()

    if benchmark_mode == BenchmarkMode.CUDA_EVENTS:
        ms, min_ms, max_ms = triton.testing.do_bench(
            call_func_under_test,
            quantiles=quantiles,
            warmup=warmup_rep,
            rep=bench_rep,
        )
    elif benchmark_mode == BenchmarkMode.TORCH_COMPILE:
        ms, min_ms, max_ms = triton.testing.do_bench(
            compiled_fn,
            quantiles=quantiles,
            warmup=warmup_rep,
            rep=bench_rep,
        )
    elif benchmark_mode == BenchmarkMode.CUDA_GRAPHS:
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            call_func_under_test,
            quantiles=quantiles,
            rep=bench_rep,
        )
    elif benchmark_mode == BenchmarkMode.END2END:
        ms, min_ms, max_ms = end2end_bench(
            call_func_under_test,
            quantiles=quantiles,
            warmup=warmup_rep,
            rep=bench_rep,
        )
    else:
        ms = float("nan")
        min_ms = float("nan")
        max_ms = float("nan")
    return ms, min_ms, max_ms


def create_dir_if_not_exist_recursive(path, mode=0o777):
    norm_path = os.path.normpath(path)
    paths_l = norm_path.split(os.sep)
    path_walked = f"{os.sep}"
    for p in paths_l:
        if len(p) == 0:
            continue
        path_walked = os.path.join(path_walked, p)
        create_dir_if_not_exist(path_walked, mode)


def create_dir_if_not_exist(path, mode=0o777):
    if not os.path.exists(path):
        os.mkdir(path)
        try:
            os.chmod(path, mode)
        except PermissionError as e:
            print(f"can't set permission of directory {path}: {e}")


def write_df_and_chmod(df, filename, mode=0o777):
    df.to_csv(filename, sep="\t", encoding="utf-8")
    try:
        os.chmod(filename, mode)
    except PermissionError as e:
        print(f"can't set permission of file {filename}: {e}")


if __name__ == "__main__":
    if os.environ.get("TRITON_BACKEND_PDB", "0") == "1":
        import debugpy

        host_addr = os.environ.get("TRITON_BACKEND_DEBUG_ADDR", "0.0.0.0")
        pdb_port = int(os.environ.get("TRITON_BACKEND_DEBUG_PORT", "5679"))
        debugpy.listen((host_addr, pdb_port))
        print(f"[debugpy] listening at {host_addr}:{pdb_port}; wait for client...\n")
        debugpy.wait_for_client()

    cuda_version = get_runtime_label()
    print(
        f"\nRunning on {gpu_name} with Triton {triton.__version__} using {cuda_version}...\n"
    )

    print(
        f"Test setup:\n\tIMPLEMENATION_UT: {IMPLEMENTATION_UT}\n\tMAX_VALUES: {MAX_VALUES}\n\tBENCHMARK_MODES: {BENCHMARK_MODES}"
    )

    global_pds = {}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if STORE_TEST_RESULT_PATH is not None:
        gpu_path = os.path.join(os.path.abspath(STORE_TEST_RESULT_PATH), gpu_name)
        gloabl_pd_file_prefix = os.path.join(gpu_path, timestamp)
        create_dir_if_not_exist_recursive(gloabl_pd_file_prefix)
    else:
        print("STORE_TEST_RESULT_PATH is not set; results will not be saved")
        gloabl_pd_file_prefix = None

    if do_benchmarks:
        pytest.do_benchmarks = do_benchmarks
        pytest.global_pds = global_pds
        pytest.global_pd_file_prefix = gloabl_pd_file_prefix

    test_filters = []
    start_time = datetime.now()
    # Get arguments to pass to pytest
    if len(sys.argv) >= 1:
        args = [__file__]
        filter_args = ""
        for ca in sys.argv[1:]:
            if ".conf" in ca:
                # already processed
                continue
            if ca[0] == "-":
                args.append(ca)
            else:
                filter_args += f"{ca} or "
                test_filters.append(ca)
        if len(filter_args) > 2:
            args.append(f"-k {filter_args[:-3]}")
        # print(f"starting pytest with args: {args}")
        if len(test_filters) > 0:
            print(f"\tSelected tests: {test_filters}")
        pytest.main(args=args)
    else:
        pytest.main(args=[__file__])
    end_time = datetime.now()
    duration = end_time - start_time

    # Dump final results
    if do_benchmarks:
        for test, df in pytest.global_pds.items():
            if len(df) <= 20 or force_dump_dataframes:
                print(
                    f"\nPerformance results of test {test} (only tests without numerical error and with valid shapes, etc.):"
                )
                print(df.to_string())

        if STORE_TEST_RESULT_PATH is not None:
            for test, df in pytest.global_pds.items():
                filename = os.path.abspath(
                    f"{STORE_TEST_RESULT_PATH}/{gpu_name}/{timestamp}/{test}_final.csv"
                )
                write_df_and_chmod(df, filename)
                print(f"(stored in {filename})")

    print(
        f"\nThis test used triton version: {triton.__version__}\n"
        f"This test was executed on: {gpu_name}\n"
        f"This test used: {cuda_version}\n"
        f"This test took: {duration}"
    )
