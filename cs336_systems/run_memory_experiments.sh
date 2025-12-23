#!/bin/bash
set -e

# Directory for results
mkdir -p profiling_result

# 1. Forward Pass (FP32) - Large Model
for ctx in 128 256 512; do
    echo "Running Forward Pass (FP32) Context $ctx..."
    uv run python3 cs336_systems/benchmark.py \
        --config large \
        --batch_size 1 \
        --mode forward \
        --context_length $ctx \
        --steps 5 \
        --enable_memory_profiling \
        --memory_snapshot_file profiling_result/snapshot_large_fwd_ctx${ctx}.pickle
done

# 2. Training Step (FP32) - Large Model
for ctx in 128 256 512; do
    echo "Running Training Step (FP32) Context $ctx..."
    uv run python3 cs336_systems/benchmark.py \
        --config large \
        --batch_size 1 \
        --mode training \
        --context_length $ctx \
        --steps 5 \
        --enable_memory_profiling \
        --memory_snapshot_file profiling_result/snapshot_large_train_ctx${ctx}.pickle
done

# 3. Mixed Precision (BF16) - Forward & Training
# Comparing impact of mixed precision on large model
echo "Running Forward Pass (BF16) Context 512..."
uv run python3 cs336_systems/benchmark.py \
    --config large \
    --batch_size 1 \
    --mode forward \
    --context_length 512 \
    --mixed_precision bf16 \
    --steps 5 \
    --enable_memory_profiling \
    --memory_snapshot_file profiling_result/snapshot_large_fwd_bf16_ctx512.pickle

echo "Running Training Step (BF16) Context 512..."
uv run python3 cs336_systems/benchmark.py \
    --config large \
    --batch_size 1 \
    --mode training \
    --context_length 512 \
    --mixed_precision bf16 \
    --steps 5 \
    --enable_memory_profiling \
    --memory_snapshot_file profiling_result/snapshot_large_train_bf16_ctx512.pickle
