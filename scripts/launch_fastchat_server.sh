#!/bin/bash

#Copyright (c) Meta Platforms, Inc. and affiliates.

export OUTLINES_CACHE_DIR="/tmp/.outlines.$USER"

MODEL=${1:-Meta-Llama-3.1-8B-Instruct}
NUM_GPU=${2:-8}
PORT=${3:-9001}

python -m fastchat.serve.controller &

if curl -s http://localhost:$PORT > /dev/null; then
    echo "Port $PORT is already in use"
    exit 1
fi

python -m fastchat.serve.openai_api_server --host 0.0.0.0 --port $PORT &

export RAY_ADDRESS="local"

if [[ $MODEL == *"Llama-3"* ]]; then
    # disables kvcache: https://github.com/vllm-project/vllm/issues/2729
    export ARGS="--gpu-memory-utilization 0.9 --max-model-len 47000 --enable-chunked-prefill False"
elif [[ $MODEL == *"Pixtral"* ]]; then
    export ARGS="--tokenizer-mode mistral"	
else
    export ARGS=""
fi

# don't put it into the background so the script don't terminate
python -m fastchat.serve.vllm_worker \
       --model-path $MODEL --num-gpus $NUM_GPU \
       --dtype float16 $ARGS
