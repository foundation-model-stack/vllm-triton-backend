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

try:
    from .flash_attn import (
        FlashAttnDecodeCaller,
        FlashAttnPrefillCaller,
        FlashAttnPrefixPrefillCaller,
    )
except ModuleNotFoundError:
    pass

try:
    from .xformers import XformersCaller
except ModuleNotFoundError:
    # print("[benchmark callers] xformers not present, skipping..")
    pass

try:
    from .vllm_cuda_v2 import VllmCudaV2Caller
    from .vllm_cuda_v1 import VllmCudaV1Caller
    from .baseline_triton import BaselineTritonCaller, BaselineTritonPrefixPrefillCaller
except ModuleNotFoundError:
    pass

from .triton_2d import Triton2dAttentionDecodeCaller, Triton2dChunkedPrefillCaller
from .triton_3d import Triton3dAttentionDecodeCaller, Triton3dAttentionPrefillCaller
from .triton_fp8 import TritonFp8Caller

try:
    from .flashinfer import FlashInferCaller
except (ModuleNotFoundError, ImportError):
    # print("[benchmark callers] flashinfer not present, skipping..")
    pass
from .fused_triton import (
    FusedTritonChunkedPrefixPrefill25dCaller,
    FusedTritonDecodeOnlyCaller,
)
from .pytorch_native import PytorchNativeAttentionPrefillCaller

from .unified_triton import (
    UnifiedTriton2dAttentionCaller,
    UnifiedTriton3dAttentionCaller,
    UnifiedTritonAutoAttentionCaller,
)
