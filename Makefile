TAG := vllm-triton-backend-$(shell id -un)
MAX_JOBS := 64

.PHONY: all build clean format dev rocm

all: build

vllm-all.tar: .git/modules/vllm/index
	cd vllm; ls -A | xargs tar --mtime='1970-01-01' -cf ../vllm-all.tar

all-git.tar: .git/index
	cd .git; ls -A | xargs tar --mtime='1970-01-01' -cf ../all-git.tar

dev: vllm-all.tar all-git.tar Dockerfile.dev
	docker build --progress=plain --build-arg MAX_JOBS=$(MAX_JOBS) . -t ${TAG} -f Dockerfile.dev
	@echo "Built docker image with tag: ${TAG}"

build: Dockerfile
	docker build --progress=plain --build-arg MAX_JOBS=$(MAX_JOBS) . -t ${TAG}
	@echo "Built docker image with tag: ${TAG}"

rocm: vllm-all.tar all-git.tar Dockerfile.rocm
	docker build --progress=plain --build-arg MAX_JOBS=$(MAX_JOBS) . -t ${TAG} -f Dockerfile.rocm
	@echo "Built docker image with tag: ${TAG}"

clean:
	rm -f vllm-all.tar all-git.tar

ifndef CI_ENABLED
format:
	python -m black scripts ibm_triton_lib third_party
else
format:
	python -m black --check --verbose scripts ibm_triton_lib third_party
endif
