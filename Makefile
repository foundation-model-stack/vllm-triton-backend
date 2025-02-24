TAG := vllm-triton-backend-$(shell id -un)
MAX_JOBS := 64

.PHONY: all build format

all: build

build: Dockerfile
	docker build --progress=plain --build-arg MAX_JOBS=$(MAX_JOBS) . -t ${TAG}
	@echo "Built docker image with tag: ${TAG}"

ifndef CI_ENABLED
format:
	python -m black scripts ibm_triton_lib third_party
else
format:
	python -m black --check --verbose scripts ibm_triton_lib third_party
endif
