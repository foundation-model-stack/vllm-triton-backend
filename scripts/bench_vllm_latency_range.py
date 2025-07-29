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

import os
import json
import sys
import torch
from datetime import datetime
from itertools import zip_longest, repeat, chain, product


# ================= SETUP

# selected_batch_sizes = [1]  # [4, 16, 32] #,128]
# selected_input_lengths = [500]  # , 1000, 1500, 2000, 4000, 8000, 16000]
# selected_output_lengths = [10, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
# selected_input_lengths = [64, 128, 512, 1024, 2048, 4096]
# selected_input_lengths = [64, 128, 512, 1024, 2048, 4096, 8192, 31500]
# selected_output_lengths = [1]
selected_batch_sizes = [1, 2, 4, 8, 16, 32, 64]
selected_input_lengths = [128]
selected_output_lengths = [32, 128, 256]

# use_cross_product = False
use_cross_product = True

warmup_iterations = 3
iterations = 5 

# =================


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


if len(sys.argv) < 5:
    print(f"Usage: {sys.argv[0]} <model_path> <tp-factor> <testcase_name> <result_path>")
    exit(-1)


gpu_name = torch.cuda.get_device_name().replace(" ", "_").replace("/", "_")

# model = "/model/llama3.1-8b/instruct/"
model = sys.argv[1]
tp = int(sys.argv[2])
testcase_name = sys.argv[3]
result_path = os.path.abspath(sys.argv[4])


timestamp_f = datetime.now().strftime("%Y-%m-%d_%H%M")

# result_dir = f"/results/{model.replace('/','-')}/{gpu_name}/{testcase_name}"
result_dir = f"{result_path}/{model.replace('/','-')}/{gpu_name}/{testcase_name}/exp_{timestamp_f}/"

# os.system(f"mkdir -p {result_dir}")
create_dir_if_not_exist_recursive(result_dir)

bench_script = "/workspace/benchmarks/benchmark_latency.py"
if not os.path.isfile(bench_script):
    bench_script = "vllm-triton-backend/vllm/benchmarks/benchmark_latency.py"
    if not os.path.isfile(bench_script):
        print(f"can't find benchmark script benchmark_latency.py")
        exit(-1)

if use_cross_product:
    zipped_lists = list(product(selected_batch_sizes, selected_input_lengths, selected_output_lengths))
else:
    max_length = max(len(selected_batch_sizes), len(selected_input_lengths), len(selected_output_lengths))
    zipped_lists = list(
        zip_longest(
            chain(selected_batch_sizes, 
                  repeat(selected_batch_sizes[-1], times=max_length-len(selected_batch_sizes))),
            chain(selected_input_lengths, 
                  repeat(selected_input_lengths[-1], times=max_length-len(selected_input_lengths))),
            chain(selected_output_lengths, 
                  repeat(selected_output_lengths[-1], times=max_length-len(selected_output_lengths))),
            fillvalue=None,
        )
    )
print(zipped_lists)

start_time = datetime.now()
for bs, il, ol in zipped_lists:
    print(
        f"====== Measuring batch_size {bs}, input length {il}, output length {ol} ====="
    )
    json_file_name = f"{result_dir}/result_bs_{bs}_il_{il}_ol_{ol}.json"
    cmd = (
        f"VLLM_USE_V1=1 python {bench_script} "
        f"--model {model} "
        f"--input-len {il} --output-len {ol} --batch-size {bs} "
        f"--output-json {json_file_name} "
        f"--num-iters-warmup {warmup_iterations} "
        f"--num-iters {iterations} "
        f"--tensor-parallel {tp} "
        f"--enable-chunked-prefill "
        f"--max-num-batched-tokens 16384 "
        # f"-O.full_cuda_graph=true"
    )
    print(cmd)
    rv = os.system(cmd)
    if rv != 0:
        print(f"benchmark command returned {rv}, stopping...")
        exit(rv)

end_time = datetime.now()
print(f"results stored in: {result_dir}")
os.system(f"ls -alh {result_dir}")
print(f"Benchmark time: {end_time-start_time}")
