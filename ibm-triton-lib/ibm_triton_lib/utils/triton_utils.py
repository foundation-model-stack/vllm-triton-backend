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

#  /*******************************************************************************
#   * Copyright 2024 IBM Corporation
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

##########################################
# Some utilities for working with triton #
##########################################

from __future__ import annotations

import builtins
import sys
import os
import time
import random
import string
import torch
import triton


# from https://github.com/triton-lang/triton/blob/main/third_party/proton/tutorials/matmul.py (also apache 2.0)
def unpack_grid(grid):
    if len(grid) == 1:
        return grid[0], 1, 1
    if len(grid) == 2:
        return grid[0], grid[1], 1
    if len(grid) == 3:
        return grid[0], grid[1], grid[2]


cuda_version = None
rocm_version = None
flag_print_debug = False


def _get_cuda_version():
    """
    Get CUDA runtime/driver version (i.e. which ptxas is used).
    This version is often different from the cuda version pytorch uses internally.

    Based on https://github.com/triton-lang/triton/blob/9d6736a501d0499348d48d192b6260338ca19da0/third_party/nvidia/backend/compiler.py#L32-L37
    """
    global cuda_version
    if cuda_version is not None:
        return cuda_version
    if "_TRITON_DEJAVU_DETERMINED_CUDA_VERSION" in os.environ:
        cuda_version = os.environ["_TRITON_DEJAVU_DETERMINED_CUDA_VERSION"]
        return cuda_version
    try:
        import subprocess
        import re

        triton_backend_dir = os.path.dirname(triton.backends.__file__)
        ptxas_path = os.path.abspath(
            os.path.join(triton_backend_dir, "nvidia/bin/ptxas")
        )

        result = subprocess.check_output(
            [ptxas_path, "--version"], stderr=subprocess.STDOUT
        )
        version = re.search(
            r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE
        )
        cuda_version = version.group(1)
    except Exception as e:
        if flag_print_debug:
            print(
                f"[triton-dejavu] determining cuda version failed with: {e}\n"
                f"using torch.version.cuda as fallback"
            )
        cuda_version = f"torch_{torch.version.cuda}"
    os.environ["_TRITON_DEJAVU_DETERMINED_CUDA_VERSION"] = cuda_version
    return cuda_version


def _get_rocm_version():
    """
    Get ROCM runtime/driver version (i.e. which rocm linker is used).
    This version is often different from the rocm version pytorch uses internally.
    """
    global rocm_version
    if rocm_version is not None:
        return rocm_version
    if "_TRITON_DEJAVU_DETERMINED_ROCM_VERSION" in os.environ:
        rocm_version = os.environ["_TRITON_DEJAVU_DETERMINED_ROCM_VERSION"]
        return rocm_version
    try:
        import subprocess
        import re

        rocm_ldd_path = triton.backends.backends["amd"].compiler.path_to_rocm_lld()
        rocm_dir = os.path.dirname(rocm_ldd_path)
        amdgpu_arch_path = os.path.abspath(os.path.join(rocm_dir, "amdgpu-arch"))

        result = subprocess.check_output(
            [amdgpu_arch_path, "--version"],
            stderr=subprocess.STDOUT,
        )
        version = re.search(
            r".*roc-(\d+\.\d+.\d+).*", result.decode("utf-8"), flags=re.MULTILINE
        )
        rocm_version = version.group(1)
    except Exception as e:
        if flag_print_debug:
            print(
                f"[triton-dejavu] determining rocm version failed with: {e}\n"
                f"using torch.version.hip as fallback"
            )
        rocm_version = f"torch_{torch.version.hip}"
    os.environ["_TRITON_DEJAVU_DETERMINED_ROCM_VERSION"] = rocm_version
    return rocm_version



def get_runtime_label():
    if torch.version.hip:
        return f"rocm_{_get_rocm_version()}"
    return f"cuda_{_get_cuda_version()}"
