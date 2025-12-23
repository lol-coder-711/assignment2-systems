## 1. Memory Profiling Analysis (Sawtooth Pattern)

**Observation:**
In a "Forward Only" pass on a high-end GPU (RTX A4500) using BF16, the memory timeline exhibits a symmetric "Sawtooth" pattern where the "Ramp Up" (Allocation) and "Ramp Down" (Deallocation) phases have nearly identical durations (Slope ≈ 1).

**Explanation:**
This symmetry indicates that the workload is **CPU/Allocator Bound** rather than GPU Compute Bound.
1.  **Rise Phase:** Corresponds to the sequential allocation and kernel launch of layers.
    *   Time = `GPU_Compute + Allocator_Overhead`.
    *   On a fast GPU (A4500) with BF16, `GPU_Compute` is negligible.
    *   Therefore, `Rise_Time ≈ Allocator_Overhead`.
2.  **Fall Phase:** Corresponds to the sequential destruction of the computation graph (freeing tensors).
    *   Time = `Deallocator_Overhead`.
    *   This is a CPU-side serial process ($O(N)$ graph nodes).
3.  **Symmetry:** Since the overhead of allocating a tensor is roughly similar to (or slightly slower than) freeing it, the rise and fall times are matched, creating the symmetric triangle.

*Contrast:* On a slower GPU (T4), compute time dominates the rise phase, leading to a classic asymmetric pattern (Slow Rise, Fast Fall).

---

## 2. Problem (pytorch_attention)

### (a) Benchmarking Naive Attention
**Configuration:** Batch=8, Single Head (d_head varies), FP32.

| d_head | seq_len | Fwd (ms) | Bwd (ms) | Peak Mem (GB) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **16** | 256 | 0.49 | 1.25 | 0.03 | OK |
| **16** | 1024 | 0.94 | 2.60 | 0.22 | OK |
| **16** | 4096 | 12.80 | 32.94 | 3.25 | OK |
| **16** | 8192 | 51.12 | 130.75 | 12.93 | OK |
| **16** | **16384** | - | - | **OOM** | **OOM** |
| | | | | | |
| **32** | 8192 | 52.88 | 133.32 | 12.95 | OK |
| **32** | 16384 | - | - | OOM | OOM |
| | | | | | |
| **64** | 8192 | 55.52 | 136.87 | 13.00 | OK |
| **64** | 16384 | - | - | OOM | OOM |

### Analysis
**OOM Point:**
The implementation consistently runs **Out Of Memory** at **Sequence Length = 16,384** on a 20GB GPU.

**Memory Accounting (for S=16384):**
The memory usage is dominated by the materialized **Attention Score Matrix** ($S \times S$), which must be saved for the backward pass.
*   Shape: $(B, S, S) = (8, 16384, 16384)$.
*   Elements: $2,147,483,648$.
*   Size (FP32): $2.147 \text{ billion} \times 4 \text{ bytes} \approx \textbf{8.59 GB}$.

During the forward pass, we need:
1.  **Saved Activation (Weights)**: ~8.59 GB.
2.  **Temp Buffer (Scores)**: ~8.59 GB.
3.  **Backward Gradients**: ~8.59 GB.
**Total Peak Requirement** > 20 GB.

**Memory Scaling:**
The memory cost scales **quadratically ($O(S^2)$)**. Optimization requires **FlashAttention** (or Memory Efficient Attention), which uses tiling to compute the output without materializing the full $S \times S$ matrix, reducing memory complexity to linear $O(S)$.

---

## 3. Problem (torch_compile)

### (a) Compiled Attention Benchmark
**Configuration:** d_head=16.

| seq_len | Van Fwd (ms) | Van Bwd (ms) | **Cmp Fwd (ms)** | **Cmp Bwd (ms)** | Speedup (Fwd) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 256 | 0.57 | 1.29 | 0.38 | 0.76 | 1.5x |
| 1024 | 0.91 | 2.60 | 0.48 | 1.61 | 1.9x |
| 4096 | 12.84 | 33.03 | **6.14** | **16.96** | **2.1x** |
| 8192 | 51.33 | 131.25 | **28.94** | **69.79** | **1.8x** |

**Conclusion:** `torch.compile` provides a consistent **1.8x - 2.1x** speedup for the naive attention kernel by fusing standard pointwise operations (element-wise scaling, masking, softmax). However, it does not solve the quadratic memory bottleneck.

### (b) End-to-End Model Compile
**Configuration:** Large Model (`d_model=1280`, `layers=36`), `Context=512`, `Batch=1`.

| Metric | Vanilla (s) | Compiled (s) | Speedup |
| :--- | :--- | :--- | :--- |
| **Forward Pass** | 0.1294 | 0.1108 | **1.17x** |
| **Training Step** | 0.5564 | 0.5029 | **1.11x** |

**Conclusion:**
Compiling the full model yields a modest **11-17% speedup**. The gain is smaller than the isolated attention kernel because the large model's runtime is dominated by large Matrix Multiplications (Linear layers), which are already highly optimized by cuBLAS and gain minimal benefit from compiler fusion compared to memory-bound element-wise operations.
