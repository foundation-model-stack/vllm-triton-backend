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


def register():
    """Register the triton attention platform."""

    VLLM_USE_V1 = int(os.environ.get("VLLM_USE_V1", "0"))

    # backend only works with v0 currently
    if VLLM_USE_V1:
        return None
    else:
        return "ibm_triton_lib.backend.platform.TritonPlatform"
