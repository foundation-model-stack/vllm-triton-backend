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

# create fake module if triton-dejavu is not present
#  remove ASAP
try:
    import triton_dejavu
except ImportError:
    import sys

    class Fake_autotuner(object):

        def __init__(self, *args, **ignore_args):
            pass

        def __call__(self, *args, **kwds):
            pass

        def run(self, *args, **kwargs):
            print(
                "ERROR: triton-dejavu is called while not being installed. Please install triton-dejavu!"
            )
            raise ImportError

        def __getitem__(self, grid):
            print(
                "ERROR: triton-dejavu is called while not being installed. Please install triton-dejavu!"
            )
            raise ImportError

    class Fake_triton_dejavu(object):

        def autotune(*args, **kwargs):
            fake_decorator = lambda fn: Fake_autotuner(fn)
            return fake_decorator

        @staticmethod
        def ConfigSpace(
            kwargs_with_lists,
            kwarg_conditions=None,
            pre_hook=None,
            **configuration_args,
        ):
            pass

    sys.modules["triton_dejavu"] = Fake_triton_dejavu
    print(
        "WARNING: Created fake module to work-around missing triton-dejavu module. If you don't expect this warning, this is likely to become an error."
    )

from .triton_flash_attention import (
    triton_wrapper_forward_prefill as prefill_flash_attention,
)

from .triton_unified_attention import unified_attention

from .mamba_ssm import selective_state_update
