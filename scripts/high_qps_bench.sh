#!/bin/bash

#  uv pip install pandas datasets numpy

# MODEL=meta-llama/Llama-3.1-8B-Instruct
# MODEL=/net/storage149/autofs/css22/nmg/models/hf/ibm-granite/granite-4.0-tiny-preview/main/
MODEL=/net/storage149/autofs/css22/nmg/models/cos/1bfc857/fmaas-integration-tests/models/granite-4_0-small-base-pipecleaner-hf
# MODEL=/net/storage149/autofs/css22/nmg/models/hf/ibm-ai-platform/Bamba-9B-v1/main/
REQUEST_RATES=(20 20 20)
TOTAL_SECONDS=120

for REQUEST_RATE in "${REQUEST_RATES[@]}";
do
        NUM_PROMPTS=$(($TOTAL_SECONDS * $REQUEST_RATE))
        echo ""
        echo "===== RUNNING $MODEL FOR $NUM_PROMPTS PROMPTS WITH $REQUEST_RATE QPS ====="
        echo ""
        python3 vllm-triton-backend/vllm/benchmarks/benchmark_serving.py \
                --model $MODEL \
                --dataset-name random \
                --ignore-eos \
                --num-prompts $NUM_PROMPTS \
                --request-rate $REQUEST_RATE \
                --port 8803 \
        ;
done;
