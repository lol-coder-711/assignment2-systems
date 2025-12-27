#!/bin/bash
# Script to run DDP benchmarks with nsys profiling
# Usage: ./run_benchmark_with_nsys.sh

# Configuration
# Format: "method" or "bucketed:size_mb"
METHODS=("naive" "flatten" "overlap" "bucketed:1" "bucketed:10" "bucketed:100" "bucketed:1000")
WORLD_SIZE=2
OUTPUT_DIR="./nsys_profiles"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting DDP benchmarking..."
echo "Timestamp: $TIMESTAMP"
echo "Methods: ${METHODS[@]}"
echo "World size: $WORLD_SIZE"
echo ""

for method in "${METHODS[@]}"; do
    # Parse method:bucket_size format
    if [[ "$method" == bucketed:* ]]; then
        BUCKET_SIZE="${method#bucketed:}"
        METHOD_NAME="bucketed"
        DISPLAY_NAME="Bucketed ${BUCKET_SIZE}MB"
        OUTPUT_FILE="${OUTPUT_DIR}/ddp_bucketed_${BUCKET_SIZE}mb_ws${WORLD_SIZE}_${TIMESTAMP}.nsys-rep"
        EXTRA_ARGS="--bucket-size-mb $BUCKET_SIZE"
    else
        METHOD_NAME="$method"
        DISPLAY_NAME="$method"
        OUTPUT_FILE="${OUTPUT_DIR}/ddp_${method}_ws${WORLD_SIZE}_${TIMESTAMP}.nsys-rep"
        EXTRA_ARGS=""
    fi
    
    echo "========================================"
    echo "Running: $DISPLAY_NAME (World Size: $WORLD_SIZE)"
    echo "========================================"
    
    if command -v nsys &> /dev/null && command -v nvidia-smi &> /dev/null; then
        # CUDA environment with nsys available
        nsys profile \
            --trace-fork-before-exec=false \
            --flush-on-cudaprofilerstop=false \
            --capture-range=cudaProfilerApi \
            --wait=all \
            --sample=none \
            --trace=cuda,nvtx \
            --output="$OUTPUT_FILE" \
            --force-overwrite=true \
            python cs336_systems/dist_training/benchmark_ddp.py \
                --method "$METHOD_NAME" \
                $EXTRA_ARGS \
                --world-size "$WORLD_SIZE" \
                --enable-profiling || true
        
        sync
        sleep 2
        
        if [ -f "${OUTPUT_FILE}" ]; then
            FILE_SIZE=$(stat -f%z "${OUTPUT_FILE}" 2>/dev/null || stat -c%s "${OUTPUT_FILE}" 2>/dev/null)
            echo "✓ Profile saved: ${OUTPUT_FILE} (${FILE_SIZE} bytes)"
        fi
    else
        # CPU/MPS - run without profiling
        python cs336_systems/dist_training/benchmark_ddp.py \
            --method "$METHOD_NAME" \
            $EXTRA_ARGS \
            --world-size "$WORLD_SIZE"
    fi
    
    echo ""
done

echo "========================================"
echo "All benchmarks completed!"
echo "Timestamp: $TIMESTAMP"
echo "========================================"

if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "Generated profiles:"
    ls -lh "$OUTPUT_DIR"/*${TIMESTAMP}* 2>/dev/null || echo "No profiles found for this run"
fi
