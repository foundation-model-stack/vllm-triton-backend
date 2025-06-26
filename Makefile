TAG := vllm-triton-backend-$(shell id -un)
MAX_JOBS := 64

SHELL := /bin/bash

.PHONY: all build clean format dev rocm rocm-upstream pyupdate nightly bm-rocm spelling

all: build

vllm-all.tar: .git/modules/vllm/index
	@# cd vllm; git ls-files | xargs tar --mtime='1970-01-01' -cf ../vllm-all.tar
	cd vllm; git ls-files > .to-compress; tar -T .to-compress --mtime='1970-01-01 00:00:00' -W -cf ../vllm-all.tar; rm .to-compress

all-git.tar: .git/HEAD
	cd .git; ls -A | xargs tar --mtime='1970-01-01' -cf ../all-git.tar

ShareGPT_V3_unfiltered_cleaned_split.json:
	wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json


dev: vllm-all.tar all-git.tar Dockerfile ShareGPT_V3_unfiltered_cleaned_split.json
	docker build --progress=plain --build-arg MAX_JOBS=$(MAX_JOBS) --build-arg VLLM_SOURCE=custom . -t ${TAG} 
	@echo "Built docker image with tag: ${TAG}"

nightly: Dockerfile.hub ShareGPT_V3_unfiltered_cleaned_split.json
	docker build --progress=plain --build-arg MAX_JOBS=$(MAX_JOBS) . -t ${TAG}-nightly -f Dockerfile.hub 
	@echo "Built docker image with tag: ${TAG}-nightly"

pyupdate: Dockerfile ShareGPT_V3_unfiltered_cleaned_split.json
	@echo "This build does only python updates, leaving vllm-all.tar all-git.tar (i.e. the vllm csrc) untouched!"
	docker build --progress=plain --build-arg MAX_JOBS=$(MAX_JOBS) --build-arg VLLM_SOURCE=custom . -t ${TAG} 
	@echo "Built docker image with tag: ${TAG}"

build: Dockerfile ShareGPT_V3_unfiltered_cleaned_split.json
	docker build --progress=plain --build-arg MAX_JOBS=$(MAX_JOBS) . -t ${TAG}
	@echo "Built docker image with tag: ${TAG}"

rocm: Dockerfile.rocm vllm-all.tar all-git.tar ShareGPT_V3_unfiltered_cleaned_split.json
	docker build --progress=plain --build-arg MAX_JOBS=$(MAX_JOBS) --build-arg VLLM_SOURCE=submodule . -t ${TAG} -f Dockerfile.rocm
	@echo "Built docker image with tag: ${TAG}"

# bare metal
vllm/venv_rocm:
	@#cd vllm && python3 -m venv ./venv_rocm
	cd vllm && uv venv venv_rocm --python 3.12

bm-rocm: | vllm/venv_rocm
	export VLLM_TARGET_DEVICE=rocm
	cd vllm && source ./venv_rocm/bin/activate && uv pip install -r requirements/rocm-build.txt && uv pip install -e . --no-build-isolation

vllm/venv_cuda:
	cd vllm && uv venv venv_cuda --python 3.12

bm-cuda: | vllm/venv_cuda
	cd vllm && source ./venv_cuda/bin/activate && VLLM_USE_PRECOMPILED=1 uv pip install --editable .


clean:
	rm -f vllm-all.tar all-git.tar ShareGPT_V3_unfiltered_cleaned_split.json

ifndef CI_ENABLED
format:
	python -m black scripts ibm-triton-lib third_party
else
format:
	python -m black --check --verbose scripts ibm-triton-lib third_party
endif

spelling:
	codespell ./ibm-triton-lib ./triton-dejavu ./scripts

