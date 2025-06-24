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


from benchmark import (
    test_decode_attention,
    create_dir_if_not_exist,
    create_dir_if_not_exist_recursive,
    write_df_and_chmod,
    get_runtime_label,
    Implementation,
    BenchmarkMode,
    impl_translate,
    get_gpu_label,
    method_translate,
)

import re
import string


# from https://github.com/pytorch/pytorch/issues/121219#issuecomment-2722329465
def clean_names_in_json(input_filename, output_filename):
    """
    Cleans the "name" fields in a JSON file by replacing non-ASCII characters with 'x'
    and removing internal quotation marks.

    Example of problematic input:
        {
            "name": "@"�sP(0): flat_tensor"
        }
    """
    with open(input_filename, "r", encoding="utf-8", errors="replace") as file:
        content = file.read()

        # Decode Unicode escape sequences
        content = content.encode().decode("unicode_escape")

        # Regex to find "name": "<value>"
        def replace_non_ascii_and_quotes(match):
            name = match.group(1)
            visible_printable = "".join(
                c for c in string.printable if c not in "\t\n\r\x0b\x0c}{"
            )
            cleaned_name = "".join(c if c in visible_printable else "x" for c in name)
            cleaned_name = cleaned_name.replace('"', "y")  # Replace internal quotes
            return f'"name": "{cleaned_name}"'

        # Apply regex to clean names
        cleaned_content = re.sub(
            r'"name": "([\s\S]*?)"(?=, |\}|\s*\})',
            replace_non_ascii_and_quotes,
            content,
            flags=re.DOTALL,
        )

    # Write the cleaned JSON data to a new file
    with open(output_filename, "w", encoding="utf-8") as outfile:
        outfile.write(cleaned_content)


device = "cuda:0"
gpu_name = get_gpu_label()

do_benchmarks = True
quantiles = [0.5, 0.2, 0.8]
debug_flag = os.getenv("TRITON_BACKEND_DEBUG") == "1"


# DTYPES = [torch.half, torch.bfloat16, torch.float]
DTYPES = [torch.float16]
SEEDS = [0]
MAX_VALUES = [1.0]
STORE_TEST_RESULT_PATH = os.environ.get("STORE_TEST_RESULT_PATH", None)
# HEAD_SIZES_FLASH = [32, 64, 128]  # only powers of 2!
HEAD_SIZES = [128]  # only powers of 2! for llama2 & 3
# head_size * head_numbers = hidden_size

# order:  num_query_heads, num_kv_heads
# NUM_HEADS = [(32, 32), (32, 8)]
NUM_HEADS = [(32, 8)]

# BLOCK_SIZES = [8, 16, 32]
BLOCK_SIZES = [16]
# NUM_BLOCKS = [8, 16, 32]
NUM_BLOCKS = [4321]  # "arbitrary values for testing..."

# options most likely not used...but keep for now?
CAUSAL_FLASH = [True]  # vLLM only needs causal=True

PROMPT_PATTERNS = []
# PROMPT_PATTERNS.append([1.0])
# PROMPT_PATTERNS.append([1.0, 0.4, 0.5, 1.0, 0.2])
PROMPT_PATTERNS.append([0.1, 0.4, 0.5, 1.0, 0.2])

BATCH_SIZES = [4]
SEQUENCE_LENGTHS = [128]


MY_IUT = [
    e for e in os.environ.get("MY_IUT", "").split(",") if len(e) > 0
]  # my implementations under test (IUT)
MY_METHODS = [e for e in os.environ.get("MY_METHODS", "").split(",") if len(e) > 0]

if len(MY_IUT) > 0:
    IMPLEMENTATION_UT = []
    for ci_value in MY_IUT:
        IMPLEMENTATION_UT.append(Implementation(impl_translate[ci_value]))
if len(MY_METHODS) > 0:
    BENCHMARK_MODES = []
    for cb_value in MY_METHODS:
        BENCHMARK_MODES.append(BenchmarkMode(method_translate[cb_value]))


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
        f"Test setup:\n\tIMPLEMENATION_UT: {IMPLEMENTATION_UT}\n\tBENCHMARK_MODES: {BENCHMARK_MODES}"
    )

    global_pds = {}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if STORE_TEST_RESULT_PATH is not None:
        gpu_path = os.path.join(STORE_TEST_RESULT_PATH, gpu_name)
        gloabl_pd_file_prefix = os.path.join(gpu_path, timestamp)
        create_dir_if_not_exist_recursive(gloabl_pd_file_prefix)
    else:
        print("STORE_TEST_RESULT_PATH is not set; results will not be saved")
        gloabl_pd_file_prefix = None

    global_pds = {}
    start_time = datetime.now()

    for bench_m in BENCHMARK_MODES:
        for impl in IMPLEMENTATION_UT:
            prof_filename = f"{gloabl_pd_file_prefix}/trace_{bench_m}-{impl}.json"
            test_decode_attention(
                None,
                None,
                BATCH_SIZES[0],
                NUM_HEADS[0],
                SEQUENCE_LENGTHS[0],
                HEAD_SIZES[0],
                BLOCK_SIZES[0],
                NUM_BLOCKS[0],
                PROMPT_PATTERNS[0],
                DTYPES[0],
                SEEDS[0],
                impl,
                MAX_VALUES[0],
                bench_m,
                overwrite_df=global_pds,
                df_file_prefix=gloabl_pd_file_prefix,
                torch_profiling=True,
                prof_filename=f"{prof_filename}-broken",
            )
            clean_names_in_json(f"{prof_filename}-broken", prof_filename)
            print(f"profile stored in: {os.path.abspath(prof_filename)}")

    end_time = datetime.now()
    duration = end_time - start_time

    # Dump final results
    for test, df in global_pds.items():
        if len(df) <= 20:
            print(
                f"\nPerformance results of test {test} (only tests without numerical error and with valid shapes, etc.):"
            )
            print(df.to_string())

    if STORE_TEST_RESULT_PATH is not None:
        for test, df in global_pds.items():
            filename = os.path.abspath(f"{gloabl_pd_file_prefix}/{test}_final.csv")
            write_df_and_chmod(df, filename)
            print(f"(stored in {filename})")
        print(f"Torch profile traces stored in {gloabl_pd_file_prefix}/.")

    print(
        f"\nThis test used triton version: {triton.__version__}\n"
        f"This test was executed on: {gpu_name}\n"
        f"This test used: {cuda_version}\n"
        f"This test took: {duration}"
    )
