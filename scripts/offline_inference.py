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
os.environ["NGL_VLLM_USE_TRITON_BACKEND"] = "1"
# to use triton paged attention within triton backend
os.environ["NGL_VLLM_USE_TRITON_PAGED_ATTN"] = "1"
# to use all triton kernels (t.b.c.)
# os.environ['NGL_VLLM_USE_TRITON_KERNELS'] = '1'

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory


llm = LLM(
    model="/models/llama2-7b/base/",
    # model="/models/llama3-8b/base/",
    # tokenizer="/models/llama3-8b/base/",
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
    "Provide a list of instructions for preparing chicken soup for a family "
    "of four.",
    "Skating and cross country skiing technique differ in",
]


print(f"Inference with {len(prompts)} prompts...")
t0 = time.time()
outputs = llm.generate(prompts, sampling_params)
t1 = time.time()

print(f"inference time: {t1-t0:.5f}s")

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
