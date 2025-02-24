## Global Args #################################################################
ARG BASE_UBI_IMAGE_TAG=9.4
ARG PYTHON_VERSION=3.12
ARG MAX_JOBS=64
ARG PIP_VLLM_VERSION=0.7.2

## Base Layer ##################################################################
FROM registry.access.redhat.com/ubi9/ubi-minimal:${BASE_UBI_IMAGE_TAG} AS base
ARG PYTHON_VERSION
ENV PYTHON_VERSION=${PYTHON_VERSION}
RUN microdnf -y update && microdnf install -y \
    python${PYTHON_VERSION}-devel python${PYTHON_VERSION}-pip python${PYTHON_VERSION}-wheel \
    gzip tar git\
    && microdnf clean all

WORKDIR /workspace

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

## Common Builder #################################################################
FROM base AS common-builder
ARG PYTHON_VERSION

ENV VIRTUAL_ENV=/opt/build
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# create new venv to build vllm
RUN python${PYTHON_VERSION} -m venv $VIRTUAL_ENV \
    && pip install --no-cache -U pip wheel uv

# install compiler cache to speed up compilation leveraging local or remote caching
# git is required for the cutlass kernels
RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm && rpm -ql epel-release && microdnf install -y ccache && microdnf clean all

## Triton Builder #################################################################
FROM common-builder AS triton-builder

# Triton build deps
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install ninja cmake wheel pybind11 setuptools

COPY triton triton

WORKDIR /workspace/triton/python

# needed to build triton
RUN microdnf install -y zlib-devel gcc gcc-c++ \
    && microdnf clean all

# Build Triton
ENV TRITON_BUILD_WITH_CCACHE=true
ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    python3 setup.py bdist_wheel --dist-dir=/workspace/

## Runtime #################################################################
FROM base AS runtime

ENV VIRTUAL_ENV=/opt/runtime
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# create new venv to build vllm
RUN python${PYTHON_VERSION} -m venv $VIRTUAL_ENV \
    && pip install --no-cache -U pip wheel uv

# swig is required by triton-dejavu (SMAC optimizer)
# SWIG rpm not available for RHEL9
RUN microdnf install -y wget tar zlib-devel automake g++ && microdnf clean all
RUN wget https://downloads.sourceforge.net/project/swig/swig/swig-3.0.12/swig-3.0.12.tar.gz && \
    tar -xzf swig-3.0.12.tar.gz && \
    cd swig-3.0.12 && \
    bash autogen.sh && \
    wget https://downloads.sourceforge.net/project/pcre/pcre/8.45/pcre-8.45.tar.gz && \
    bash Tools/pcre-build.sh && \
    bash ./configure && \
    make && \
    make install

WORKDIR /workspace

# Install vllm
ARG PIP_VLLM_VERSION
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install vllm==${PIP_VLLM_VERSION}

# to avaoid incompatibility with our custom triton build
#  see also https://github.com/vllm-project/vllm/issues/12219
RUN uv pip install -U torch>=2.6 torchvision>=2.6 torchaudio>=2.6

# Install Triton (will replace version that vllm/pytorch installed)
COPY --from=triton-builder /workspace/*.whl .
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install triton-*.whl

# force using the python venv's cuda runtime libraries
ENV LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/nvidia/nvtx/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/nvidia/cuda_cupti/lib:${LD_LIBRARY_PATH}"

# copy requirements before to avoid reinstall
COPY triton-dejavu/requirements-opt.txt dejavu-requirements-opt.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r dejavu-requirements-opt.txt

    # dejavu
COPY triton-dejavu triton-dejavu
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install ./triton-dejavu/

## Benchmarking #################################################################
FROM runtime AS benchmark

WORKDIR /workspace

RUN microdnf install -y git nano gcc vim \
    && microdnf clean all

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install pytest llnl-hatchet debugpy

# Install FlashInfer
RUN PYTHON_VERSION_STR=$(echo ${PYTHON_VERSION} | sed 's/\.//g') && \
    echo "export PYTHON_VERSION_STR=${PYTHON_VERSION_STR}" >> /etc/environment

RUN --mount=type=cache,target=/root/.cache/pip \
    . /etc/environment && \
    python3 -m pip install https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.6/flashinfer-0.1.6+cu121torch2.4-cp${PYTHON_VERSION_STR}-cp${PYTHON_VERSION_STR}-linux_x86_64.whl

RUN ln -s ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12  ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/nvidia/cuda_cupti/lib/libcupti.so

# Install IBM kernels and vllm plugin
#  must be after vllm!
# (here for now, since they should change frequently)
COPY ibm_triton_lib ibm_triton_lib/ibm_triton_lib
COPY setup.py ibm_triton_lib
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install ./ibm_triton_lib \
    && rm -rf ibm_triton_lib


ENV STORE_TEST_RESULT_PATH=/results

# Copy thid-party kernels and insert into path
COPY third_party third_party
ENV PYTHONPATH /workspace

# see https://github.com/IBM/triton-dejavu?tab=readme-ov-file#environment-variables
ENV TRITON_PRINT_AUTOTUNING=1
ENV TRITON_DEJAVU_DEBUG=1
# set as default
ENV TRITON_DEJAVU_STORAGE=/storage
ENV NGL_EXP_FALLBACK=next
# ENV TRITON_DEJAVU_USE_ONLY_RESTORED=0
ENV TRITON_DEJAVU_FORCE_FALLBACK=1
ENV TRITON_DEJAVU_TAG='default'
ENV TRITON_DEJAVU_HASH_SEARCH_PARAMS=0

# open debugpy port
EXPOSE 5679

ENTRYPOINT ["python"]
