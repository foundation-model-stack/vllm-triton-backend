TAG := vllm-triton-backend-$(shell id -un)
MAX_JOBS := 64

.PHONY: all build clean format dev

all: build

vllm-all.tar: .git/modules/vllm/index
	cd vllm; ls -A | xargs tar --mtime='1970-01-01' -cf ../vllm-all.tar

all-git.tar: .git/HEAD
	cd .git; ls -A | xargs tar --mtime='1970-01-01' -cf ../all-git.tar

ShareGPT_V3_unfiltered_cleaned_split.json:
	wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json


dev: vllm-all.tar all-git.tar Dockerfile ShareGPT_V3_unfiltered_cleaned_split.json
	docker build --progress=plain --build-arg MAX_JOBS=$(MAX_JOBS) --build-arg VLLM_SOURCE=custom . -t ${TAG} 
	@echo "Built docker image with tag: ${TAG}"

build: Dockerfile ShareGPT_V3_unfiltered_cleaned_split.json
	docker build --progress=plain --build-arg MAX_JOBS=$(MAX_JOBS) . -t ${TAG}
	@echo "Built docker image with tag: ${TAG}"


clean:
	rm -f vllm-all.tar all-git.tar ShareGPT_V3_unfiltered_cleaned_split.json

ifndef CI_ENABLED
format:
	python -m black scripts ibm-triton-lib third_party
else
format:
	python -m black --check --verbose scripts ibm-triton-lib third_party
endif
