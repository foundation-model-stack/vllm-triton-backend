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

os.environ["VLLM_PLUGINS"] = ""  

os.environ["VLLM_USE_V1"] = "1"
# os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"
os.environ["VLLM_ATTENTION_BACKEND"] = "EXPERIMENTAL_HELION_ATTN"

# enable torch profiler, can also be set on cmd line
# enable_profiling = True
enable_profiling = False

if enable_profiling:
    os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_torch_profile"


if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    llm = LLM(
        # model="meta-llama/Llama-3.1-8B-Instruct",
        model=f"{os.environ["MY_MODEL_PATH"]}",
        # max_model_len=2048,
        # enforce_eager=True,
        # enable_prefix_caching=False,
    )

    # batch_size = 32
    max_tokens = 20

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    # ignore_eos=True)

    prompts = [
        # "some long sequence that prevents helion from crashing....juts adding here: Repeat after me: Tiling! some long sequence that prevents helion from crashing....juts adding here: Repeat after me: Tiling!",
        "Zurich is a beautiful city with",
        "San Francisco is a large city with",
        "some long sequence that prevents helion from crashing....juts adding here: Repeat after me: Tiling!",
        # "Provide a list of instructions for preparing chicken soup for a family "
        # "of four.",
        # "Skating and cross country skiing technique differ in",
    ]

    print(
        f"SETUP: vllm backend: {os.environ['VLLM_ATTENTION_BACKEND']}  "
    )
    print(f"Inference with {len(prompts)} prompts...")
    if enable_profiling:
        llm.start_profile()
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    # outputs = []
    # for prompt in prompts:
    #     outputs.append(llm.generate(prompt, sampling_params))

    if enable_profiling:
        llm.stop_profile()
    t1 = time.time()

    print(f"inference time: {t1-t0:.5f}s")

    for output in outputs:
        # output = output[0]  # in case of loop above
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.
    time.sleep(10)
