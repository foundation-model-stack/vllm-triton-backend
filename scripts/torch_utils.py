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

import torch
import timeit


def get_gpu_label():
    gpu_name = torch.cuda.get_device_name().replace(" ", "_").replace("/", "_")
    return gpu_name


def pytorch_timer() -> float:
    # based on: https://github.com/pytorch/pytorch/blob/main/torch/utils/benchmark/utils/timer.py
    torch.cuda.synchronize()
    return timeit.default_timer()


# TODO: move to triton-dejavu?
def end2end_bench(
    fn, warmup=25, rep=100, quantiles=None, return_mode="mean", n_repeat_inner=1
):
    assert return_mode in ["min", "max", "mean", "median"]
    # JIT, if necessary
    fn()
    torch.cuda.synchronize()

    # to clear L2...
    cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")
    setup_l = lambda: cache.zero_()
    torch.cuda.synchronize()
    # setup_l = lambda: cache.zero_()
    # stmt_l = lambda:  torch.cuda.synchronize(); fn(); torch.cuda.synchronize()

    timer = timeit.Timer(stmt=fn, setup=setup_l, timer=pytorch_timer)

    estimate_ms = (timer.timeit(5) / 5) * 1000
    # print(estimate_ms)
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms / n_repeat_inner))
    # print(n_warmup)
    for _ in range(n_warmup):
        # only fn, no cache clear?
        #  as done in triton.do_bench
        fn()
    # print(n_repeat)
    with torch.no_grad():
        times_f = timer.repeat(repeat=n_repeat, number=n_repeat_inner)

    times_f_ms = [float(f * 1000.0 / n_repeat_inner) for f in times_f]
    times = torch.tensor(times_f_ms, dtype=torch.float)
    del cache
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()
