# vllm-triton-backend

This repo contains:

- A Triton-only attention backend for vLLM, implemented as [vLLM platform plugin](https://docs.vllm.ai/en/latest/design/plugin_system.html), see [`ibm_triton_lib/backend`](./ibm_triton_lib/backend/). 
- New Triton kernels that implement different attention algorithms, see [`ibm_triton_lib/kernels`](./ibm_triton_lib/kernels/).
- Containerized development environment (vLLM + Triton built from source). 
- A microbenchmarking framework for evaluating their performance. 

Triton kernels require autotuning to achieve best possible performance, but naïve autotuning comes with a significant overhead at runtime. Therefore, this repository depends on [triton-dejavu](https://github.com/IBM/triton-dejavu) to reduce the overhead of autotuning to zero while still adapting triton kernels for each platform and request individually. The necessary dejavu data can be found in [`ibm_triton_lib/kernels/dejavu_data`](./ibm_triton_lib/kernels/dejavu_data/).

## How to use

This repository can be used as microbenchmark framework and as vLLM plugin. In the following, we explain how to [build our container development environment](#1-build), how to [run microbenchmarks](#2-run-microbenchmarks), and how to [run triton-only attention in vllm](#3-run-vllm-triton-only-backend).

### 1) build

To build the docker image:
```
git clone --recursive https://github.com/foundation-model-stack/vllm-triton-backend.git
cd vllm-triton-backend
make build
```

Please note that this build process installs the pre-build vllm v0.7.2. 

### 2) run microbenchmarks

To run the various benchmark:
```bash
mkdir results
chmod o+w results
docker run --gpus all -it --rm \
    -v $(pwd)/scripts:/scripts \
    -v $(pwd)/ibm_triton_lib:/opt/runtime/lib64/python3.12/site-packages/ibm_triton_lib/ \
    -v $(pwd)/results:/results \
    vllm-triton-backend-$(id -un) /scripts/benchmark.py
```
The results of the benchmark are written to the results folder. 
One can edit the benchmark scripts and the kernel code without rebuilding the container.

Since `ibm_triton_lib` is also installed as python package in the vllm-triton-backend image it can be used in python scripts with `import ibm_triton_lib`.
However, if latest version of the `ibm_triton_lib` should be used, without frequently re-building the docker image, it could be mounted in the installed directory, which is currently `/opt/runtime/lib64/python3.12/site-packages/ibm_triton_lib/`, as shown above. Similar applies for the `triton_dejavu` or `vllm` module or the `scripts` folder.

### 3) run vllm triton-only backend

#### Using our container

To run vLLM with triton-only attention backend after [building our container](#1-build):
```bash
docker run -it --rm --gpus all /path/to/models:/models vllm-triton-backend-$(id -un):latest -m vllm.entrypoints.openai.api_server --model /models/granite3.1-8b/base/

```

#### Outside/stand-alone of our environment

The Triton-only attention backend can be used within our [Docker container](#dev-environment) or outside. 
To install this plugin in any other environment:
```
git clone https://github.com/foundation-model-stack/vllm-triton-backend.git
pip install ./vllm-triton-backend
```

If using `ibm_triton_lib` outside from our container, the following needs to be taken into account:

- at least triton 3.2 is required, and therefore pytorch >= 2.6
- our plugin must be installed after vllm (see [documentation](https://docs.vllm.ai/en/latest/design/plugin_system.html))
- the vllm-triton-backend depends on [triton-dejavu](https://github.com/IBM/triton-dejavu)


