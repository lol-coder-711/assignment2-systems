#!/bin/bash
# Script to run DDP benchmarks with nsys profiling
# Usage: ./run_benchmark_with_nsys.sh

# Note: Removed 'set -e' to allow script to continue even if one method fails

# Configuration
METHODS=("naive" "flatten" "overlap")
WORLD_SIZE=2
OUTPUT_DIR="./nsys_profiles"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting DDP benchmarking with nsys profiling..."
echo "Timestamp: $TIMESTAMP"
echo "Output directory: $OUTPUT_DIR"
echo "Methods: ${METHODS[@]}"
echo "World size: $WORLD_SIZE"
echo ""

for method in "${METHODS[@]}"; do
    echo "========================================"
    echo "Running: $method (World Size: $WORLD_SIZE)"
    echo "========================================"
    
    # Generate output filename with timestamp
    OUTPUT_FILE="${OUTPUT_DIR}/ddp_${method}_ws${WORLD_SIZE}_${TIMESTAMP}.nsys-rep"
    
    # Run with nsys profiling
    # Key options explained:
    # --trace-fork-before-exec=false: Don't trace child processes created by spawn
    # --flush-on-cudaprofilerstop=false: Prevent corruption in multi-context scenarios
    # --capture-range=cudaProfilerApi: Only profile between cudaProfilerStart/Stop
    # --trace=cuda,nvtx: Capture CUDA and NVTX events
    
    if command -v nsys &> /dev/null && [[ $(uname) != "Darwin" ]]; then
        # NVIDIA GPU with nsys available
        # Key fixes for mp.spawn corruption:
        # 1. --wait=all: Wait for all child processes to finish before ending profiling
        # 2. --sample=none: Disable sampling to reduce overhead/corruption risk
        # 3. Keep --flush-on-cudaprofilerstop=false to avoid multi-context issues
        # 4. Added dist.barrier() in Python code before cudaProfilerStop
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
                --method "$method" \
                --world-size "$WORLD_SIZE" \
                --enable-profiling || true  # Continue even if nsys returns non-zero exit code
        
        # Sync and wait to ensure file is complete
        sync
        sleep 3
        
        # Verify the output file
        if [ -f "${OUTPUT_FILE}" ]; then
            FILE_SIZE=$(stat -f%z "${OUTPUT_FILE}" 2>/dev/null || stat -c%s "${OUTPUT_FILE}" 2>/dev/null)
            echo "✓ Profile saved: ${OUTPUT_FILE} (${FILE_SIZE} bytes)"
            
            # Generate MD5 checksum for verification
            if command -v md5sum &> /dev/null; then
                md5sum "${OUTPUT_FILE}" > "${OUTPUT_FILE}.md5"
            elif command -v md5 &> /dev/null; then
                md5 "${OUTPUT_FILE}" > "${OUTPUT_FILE}.md5"
            fi
        else
            echo "✗ Warning: Profile file not found: ${OUTPUT_FILE}"
        fi
    else
        # CPU/MPS or nsys not available - run without profiling
        echo "nsys not available or not on Linux, running without profiling..."
        python cs336_systems/dist_training/benchmark_ddp.py \
            --method "$method" \
            --world-size "$WORLD_SIZE"
    fi
    
    echo ""
done

echo "========================================"
echo "All benchmarks completed!"
echo "Profiles saved in: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo "========================================"

# List all generated profiles
if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "Generated profiles:"
    ls -lh "$OUTPUT_DIR"/*${TIMESTAMP}* 2>/dev/null || echo "No profiles found for this run"
fi
