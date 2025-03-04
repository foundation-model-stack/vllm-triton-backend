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


num_users_to_test = [1, 2, 4, 8, 16, 32, 64, 128]
gpu_name = torch.cuda.get_device_name().replace(" ", "_").replace("/", "_")

# model = "/model/llama3.1-8b/instruct/"
model = sys.argv[1]
model_path = f"/models/{model}/"
testcase_name = sys.argv[2]
max_rounds = 128
max_num_prompts = 1000

timestamp_f = datetime.now().strftime("%Y-%m-%d_%H%M")

# result_dir = f"/results/{model.replace('/','-')}/{gpu_name}/{testcase_name}"
result_dir = (
    f"/results/{model.replace('/','-')}/{gpu_name}/{testcase_name}/exp_{timestamp_f}/"
)

# os.system(f"mkdir -p {result_dir}")
create_dir_if_not_exist_recursive(result_dir)

for max_concurrency in num_users_to_test:
    num_prompts = (
        max_num_prompts
        if max_num_prompts // max_concurrency < max_rounds
        else int(max_rounds * max_concurrency)
    )
    cmd = (
        f"VLLM_USE_V1=1 python /workspace/benchmarks/benchmark_serving.py "
        f"--model {model_path} "
        f"--dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json "
        f"--save-result --result-dir {result_dir} --max-concurrency {max_concurrency} "
        f"--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 20,50,80 "
        f"--num-prompts {num_prompts} "
    )
    print(cmd)
    os.system(cmd)

print(f"results stored in: {result_dir}")
os.system(f"ls -alh {result_dir}")
