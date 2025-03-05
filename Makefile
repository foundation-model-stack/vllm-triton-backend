TAG := vllm-triton-backend-$(shell id -un)
MAX_JOBS := 64

.PHONY: all build clean format dev rocm rocm-upstream

all: build

vllm-all.tar: .git/modules/vllm/index
	cd vllm; ls -A | xargs tar --mtime='1970-01-01' -cf ../vllm-all.tar

all-git.tar: .git/index
	cd .git; ls -A | xargs tar --mtime='1970-01-01' -cf ../all-git.tar

dev: vllm-all.tar all-git.tar Dockerfile
	docker build --progress=plain --build-arg MAX_JOBS=$(MAX_JOBS) --build-arg VLLM_SOURCE=custom . -t ${TAG} 
	@echo "Built docker image with tag: ${TAG}"

build: Dockerfile
	docker build --progress=plain --build-arg MAX_JOBS=$(MAX_JOBS) . -t ${TAG}
	@echo "Built docker image with tag: ${TAG}"

rocm-vllm-all.tar: .git/modules/rocm_vllm/index
	cd rocm_vllm; ls -A | xargs tar --mtime='1970-01-01' -cf ../rocm-vllm-all.tar

rocm: Dockerfile.rocm rocm-vllm-all.tar all-git.tar
	docker build --progress=plain --build-arg MAX_JOBS=$(MAX_JOBS) --build-arg VLLM_SOURCE=submodule . -t ${TAG} -f Dockerfile.rocm
	@echo "Built docker image with tag: ${TAG}"

rocm-upstream: Dockerfile.rocm rocm-vllm-all.tar all-git.tar
	# echo "using https://github.com/ROCm/vllm repository; vllm submodule CURRENTLY IGNORED"
	docker build --progress=plain --build-arg MAX_JOBS=$(MAX_JOBS) --build-arg VLLM_SOURCE=upsteram . -t ${TAG} -f Dockerfile.rocm
	@echo "Built docker image with tag: ${TAG}"

clean:
	rm -f vllm-all.tar all-git.tar rocm-vllm-all.tar

ifndef CI_ENABLED
format:
	python -m black scripts ibm-triton-lib third_party
else
format:
	python -m black --check --verbose scripts ibm-triton-lib third_party
endif
