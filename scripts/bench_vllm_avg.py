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
    print(f"Usage: {sys.argv[0]} <model_path> <testcase_name> <repitions> <result_path> [<port>]")

repitions = int(sys.argv[3])
gpu_name = torch.cuda.get_device_name().replace(" ", "_").replace("/", "_")

# model = "/model/llama3.1-8b/instruct/"
model = sys.argv[1]
testcase_name = sys.argv[2]
result_path = os.path.abspath(sys.argv[4])
port = sys.argv[5] if len(sys.argv) == 6 else "8000"

# max_rounds = 128
max_rounds = 64
max_num_prompts = 1000

timestamp_f = datetime.now().strftime("%Y-%m-%d_%H%M")

# result_dir = f"/results/{model.replace('/','-')}/{gpu_name}/{testcase_name}"
result_dir = (
    f"{result_path}/{model.replace('/','-')}/{gpu_name}/{testcase_name}/exp_{timestamp_f}/"
)

# os.system(f"mkdir -p {result_dir}")
create_dir_if_not_exist_recursive(result_dir)

bench_script = "/workspace/benchmarks/benchmark_serving.py"
if not os.path.isfile(bench_script):
    bench_script = "vllm-triton-backend/vllm/benchmarks/benchmark_serving.py"
    if not os.path.isfile(bench_script):
        print(f"can't find benchmark script benchmark_serving.py")
        exit(-1)

for i in range(repitions):
    print(f"====== Repition {i} =====")
    cmd = (
        f"VLLM_USE_V1=1 python {bench_script} "
        f"--model {model} "
        f"--dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json "
        f"--save-result --result-dir {result_dir} "
        f"--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 20,50,80,99 "
        f"--port {port}"
    )
    print(cmd)
    rv = os.system(cmd)
    if rv != 0:
        print(f"benchmark command returned {rv}, stopping...")
        exit(rv)

print(f"results stored in: {result_dir}")
# os.system(f"ls -alh {result_dir}")

avg_dict = {"avg_total_token_throughput": 0, "avg_ttft": 0, "avg_itl": 0}

# Assisted by watsonx Code Assistant
for filename in os.listdir(result_dir):
    if filename.endswith('.json'):
        file_path = os.path.join(result_dir, filename)
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
                # print(f"Loaded data from {filename}:")
                avg_dict["avg_total_token_throughput"] += data["total_token_throughput"]
                # avg_dict["avg_ttft"] += data["mean_ttft_ms"]
                avg_dict["avg_ttft"] += data["median_ttft_ms"]
                avg_dict["avg_itl"] += data["median_itl_ms"]
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {filename}: {e}")

avg_dict["avg_total_token_throughput"] /= repitions
avg_dict["avg_ttft"] /= repitions
avg_dict["avg_itl"] /= repitions

print(f"\nSummary of {repitions} repitions:")
print(f"Average of total token throughputs: {avg_dict['avg_total_token_throughput']:.2f} tokens/sec")
print(f"Average of Median TTFTs:            {avg_dict['avg_ttft']:.2f} ms")
print(f"Average of Median ITLs:             {avg_dict['avg_itl']:.2f} ms")
