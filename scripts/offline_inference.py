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
import time

# to enable debug printing
# os.environ["TRITON_BACKEND_DEBUG"] = "1"

# to use triton_attn backend
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"

# enable torch profiler, can also be set on cmd line
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory


llm = LLM(
    model="/models/llama3.1-8b/instruct/",
    # max_model_len=2048,
    # enforce_eager=True,
)

# batch_size = 32
max_tokens = 20

sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
# ignore_eos=True)

prompts = [
    "Zurich is a beautiful city with",
    "San Francisco is a large city with",
    # "Provide a list of instructions for preparing chicken soup for a family "
    # "of four.",
    # "Skating and cross country skiing technique differ in",
]


print(f"Inference with {len(prompts)} prompts...")
llm.start_profile()
t0 = time.time()
# outputs = llm.generate(prompts, sampling_params)
outputs = []
for prompt in prompts:
    outputs.append(llm.generate(prompt, sampling_params))

llm.stop_profile()
t1 = time.time()

print(f"inference time: {t1-t0:.5f}s")

for output in outputs:
    output = output[0]  # in case of loop above
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# Add a buffer to wait for profiler in the background process
# (in case MP is on) to finish writing profiling output.
time.sleep(10)
