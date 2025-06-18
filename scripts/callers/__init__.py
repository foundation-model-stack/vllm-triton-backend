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


from .flash_attn import (
    FlashAttnDecodeCaller,
    FlashAttnPrefillCaller,
    FlashAttnPrefixPrefillCaller,
)

try:
    from .xformers import XformersCaller
except ModuleNotFoundError:
    # print("[benchmark callers] xformers not present, skipping..")
    pass
from .vllm_cuda_v2 import VllmCudaV2Caller
from .vllm_cuda_v1 import VllmCudaV1Caller
from .triton_2d import Triton2dAttentionDecodeCaller, Triton2dChunkedPrefillCaller
from .triton_3d import Triton3dAttentionDecodeCaller, Triton3dAttentionPrefillCaller
from .baseline_triton import BaselineTritonCaller, BaselineTritonPrefixPrefillCaller
from .triton_fp8 import TritonFp8Caller

try:
    from .flashinfer import FlashInferCaller
except ModuleNotFoundError:
    # print("[benchmark callers] flashinfer not present, skipping..")
    pass
from .fused_triton import (
    FusedTritonChunkedPrefixPrefill25dCaller,
    FusedTritonDecodeOnlyCaller,
)
