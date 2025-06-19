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

from setuptools import setup
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def find_version(filepath: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/vllm-project/vllm/blob/717f4bcea036a049e86802b3a05dd6f7cd17efc8/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


dejavu_data = package_files("ibm_triton_lib/kernels/dejavu_data/")


setup(
    name="ibm_triton_lib",
    version=find_version(os.path.join(PROJECT_ROOT, "ibm_triton_lib/__init__.py")),
    description="Triton-only backend for vLLM and Triton kernel library",
    # long_description=read(PROJECT_ROOT, "README.md"),
    # long_description_content_type="text/markdown",
    # author="Burkhard Ringlein, Tom Parnell, Jan van Lunteren, Chih Chieh Yang",
    python_requires=">=3.8",
    packages=[
        "ibm_triton_lib",
        "ibm_triton_lib.utils",
        "ibm_triton_lib.kernels",
        "ibm_triton_lib.backend",
        "ibm_triton_lib.kernels.legacy",
        "ibm_triton_lib.kernels.legacy.fused_gqa_paged",
    ],
    package_data={
        "ibm_triton_lib": dejavu_data,
    },
    include_package_data=True,
    entry_points={
        "vllm.platform_plugins": ["triton_attn = ibm_triton_lib.backend:register"]
    },
)
