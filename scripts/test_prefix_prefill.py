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
import numpy as np

os.environ["VLLM_USE_V1"] = "1"
from vllm import LLM, SamplingParams

os.environ["VLLM_V1_USE_TRITON_BACKEND"] = "0"

llm = LLM(
    model="/models/llama3.1-8b/instruct/",
    # dtype='float16',
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

doc = "Switzerland,[d] officially the Swiss Confederation,[e] is a landlocked country located in west-central Europe.[f][13] It is bordered by Italy to the south, France to the west, Germany to the north, and Austria and Liechtenstein to the east. Switzerland is geographically divided among the Swiss Plateau, the Alps and the Jura; the Alps occupy the greater part of the territory, whereas most of the country's nearly 9 million people are concentrated on the plateau, which hosts its largest cities and economic centres, including Zurich, Geneva, and Lausanne.[14]"

batch_size = 2
num_experiments = 5

docs = []

for i in range(batch_size):
    docs.append(doc)

res = []
for i in range(num_experiments):
    t0 = time.time()
    responses = llm.generate(docs, sampling_params)
    t_elap = time.time() - t0
    res.append(t_elap)

print(res)

print("t_elap: %.2f seconds" % (np.median(res)))

# print(responses)

# test rocm backend
os.environ["VLLM_V1_USE_TRITON_BACKEND"] = "1"

print("\nusing triton backend\n")
del llm

llm = LLM(
    model="/models/llama3.1-8b/instruct/",
    # dtype='float16',
)

res = []
for i in range(num_experiments):
    t0 = time.time()
    responses = llm.generate(docs, sampling_params)
    t_elap = time.time() - t0
    res.append(t_elap)

print(res)

print("t_elap: %.2f seconds" % (np.median(res)))

# print(responses)
