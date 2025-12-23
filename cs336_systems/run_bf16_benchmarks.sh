#!/bin/bash

# Array of configs
configs=("small" "medium" "large" "xl" "2.7B")
# Array of precisions
precisions=("no" "bf16")

output_file="bf16_benchmark_results.txt"
rm -f $output_file

for config in "${configs[@]}"; do
    for precision in "${precisions[@]}"; do
        echo "----------------------------------------------------------------" | tee -a $output_file
        echo "Running Config: $config | Precision: $precision" | tee -a $output_file
        echo "----------------------------------------------------------------" | tee -a $output_file
        
        # We run forward+backward
        uv run python cs336_systems/benchmark.py \
            --config "$config" \
            --mixed_precision "$precision" \
            --mode backward \
            --steps 5 \
            --warmup_steps 2 \
            --batch_size 4 \
            --context_length 128 2>&1 | tee -a $output_file
            
        echo "" | tee -a $output_file
    done
done
