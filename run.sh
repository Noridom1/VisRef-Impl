#!/bin/bash

GPU_ID=${1:-0}  # default = 0 if not provided

CUDA_VISIBLE_DEVICES=$GPU_ID python3 scripts/run_inference_single.py --index 2 --max_new_tokens 10000