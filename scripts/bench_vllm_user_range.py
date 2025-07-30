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

if len(sys.argv) < 4:
    print(f"Usage: {sys.argv[0]} <model_path> <testcase_name> <result_path>")
    exit(-1)

num_users_to_test = [1, 2, 4, 8, 16, 32, 64, 128]
gpu_name = torch.cuda.get_device_name().replace(" ", "_").replace("/", "_")

# model = "/model/llama3.1-8b/instruct/"
model = sys.argv[1]
testcase_name = sys.argv[2]
result_path = os.path.abspath(sys.argv[3])

# max_rounds = 128
max_rounds = 64
max_num_prompts = 1000

timestamp_f = datetime.now().strftime("%Y-%m-%d_%H%M")

# result_dir = (
#     f"/results/{model.replace('/','-')}/{gpu_name}/{testcase_name}/exp_{timestamp_f}/"
# )
model_print_path = model.replace('/','-')
if model_print_path[0:2] == './':
    model_print_path = model_print_path[2:]
result_dir = f"{result_path}/{model_print_path}/{gpu_name}/{testcase_name}/exp_{timestamp_f}/"

bench_script = "/workspace/benchmarks/benchmark_serving.py"
if not os.path.isfile(bench_script):
    bench_script = "./vllm-triton-backend/vllm/benchmarks/benchmark_serving.py"
    if not os.path.isfile(bench_script):
        print(f"can't find benchmark script benchmark_serving.py")
        exit(-1)

# os.system(f"mkdir -p {result_dir}")
create_dir_if_not_exist_recursive(result_dir)

start_time = datetime.now()
for max_concurrency in num_users_to_test:
    num_prompts = (
        max_num_prompts
        if max_num_prompts // max_concurrency < max_rounds
        else int(max_rounds * max_concurrency)
    )
    cmd = (
        f"VLLM_USE_V1=1 python {bench_script} "
        f"--model {model} "
        f"--dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json "
        f"--save-result --result-dir {result_dir} --max-concurrency {max_concurrency} "
        f"--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 20,50,80,99 "
        f"--num-prompts {num_prompts} "
        f"--port 8803"
    )
    print(cmd)
    rv = os.system(cmd)
    if rv != 0:
        print(f"benchmark command returned {rv}, stopping...")
        break

end_time = datetime.now()
print(f"results stored in: {result_dir}")
os.system(f"ls -alh {result_dir}")
print(f"Benchmark time: {end_time-start_time}")
