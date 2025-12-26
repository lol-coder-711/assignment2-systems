
import torch
import triton
import triton.testing
import einops
from cs336_systems.flash_attention.flash_fwd_triton import flash_fwd_kernel, FlashAttentionTriton
import pandas as pd

# 1. Implementation using einops for better readability
def standard_attention(Q, K, V, is_causal):
    # Q, K, V shape: (batch, seq, dim)
    scale = 1.0 / Q.shape[-1]**0.5
    # Einops einsum: equivalent to 'b q d, b k d -> b q k'
    S = einops.einsum(Q, K, 'b q d, b k d -> b q k') * scale
    
    if is_causal:
        # 2. Robust mask handling broadcasting over batch
        seq_len_q = Q.shape[1]
        seq_len_k = K.shape[1]
        
        q_idx = torch.arange(seq_len_q, device=Q.device)
        k_idx = torch.arange(seq_len_k, device=K.device)
        
        # Create mask: (1, Q, K) to broadcast over batch
        mask = einops.rearrange(q_idx, 'q -> 1 q 1') >= einops.rearrange(k_idx, 'k -> 1 1 k')
        S = torch.where(mask, S, float("-inf"))
        
    P = torch.softmax(S, dim=-1)
    # Einops einsum: equivalent to 'b q k, b k d -> b q d'
    O = einops.einsum(P, V, 'b q k, b k d -> b q d')
    return O

# 4. Standard Attention with torch.compile for a strong baseline
standard_attention_compiled = torch.compile(standard_attention)

def run_benchmark():
    # 3. Instruction Requirements:
    # - Seq Len: Powers of 2 from 128 to 65536
    # - Dim: Powers of 2 from 16 to 128
    # - Precision: bf16 (and fp32 for completeness)
    
    SEQ_LENS = [2**i for i in range(7, 17)] # 128, 256, ..., 65536
    DIMS = [2**i for i in range(4, 9)]      # 16, 32, 64, 128,  (up to 256 usually, but instruction says 16 to 128)
    DTYPES = [torch.bfloat16, torch.float32]
    BATCH_SIZE = 1
    IS_CAUSAL = True
    
    results = []

    print(f"{'SeqLen':<8} {'Dim':<4} {'Dtype':<8} | {'Torch(Eager)':<12} {'Torch(Compile)':<14} {'FlashAttn':<12} (Forward Latency in ms)")
    
    for seq_len in SEQ_LENS:
        for dim in DIMS:
            for dtype in DTYPES:
                # Prepare inputs
                try:
                    Q = torch.randn((BATCH_SIZE, seq_len, dim), device='cuda', dtype=dtype, requires_grad=True)
                    K = torch.randn((BATCH_SIZE, seq_len, dim), device='cuda', dtype=dtype, requires_grad=True)
                    V = torch.randn((BATCH_SIZE, seq_len, dim), device='cuda', dtype=dtype, requires_grad=True)
                    dO = torch.randn_like(Q)
                except Exception as e:
                    print(f"Skipping {seq_len}x{dim} due to OOM or error: {e}")
                    continue

                configs = [
                    ("Torch(Eager)", standard_attention),
                    ("Torch(Compile)", standard_attention_compiled),
                    ("FlashAttn", lambda q, k, v, c: FlashAttentionTriton.apply(q, k, v, c))
                ]
                
                row_data = {
                    "seq_len": seq_len,
                    "dim": dim,
                    "dtype": str(dtype).replace("torch.", "")
                }
                
                for name, func in configs:
                    # Forward Benchmark
                    try:
                        fwd_ms = triton.testing.do_bench(lambda: func(Q, K, V, IS_CAUSAL))
                    except Exception as e:
                        print(f"Benchmark failed for {name} ({dtype}): {e}")
                        fwd_ms = float('nan')
                    
                    row_data[f"{name}_fwd"] = fwd_ms
                    
                    # Backward Benchmark (only if Fwd succeeded)
                    if not  isinstance(fwd_ms, float) or fwd_ms > 0:
                        try:
                            # Run forward once to get output for backward bench
                            out = func(Q, K, V, IS_CAUSAL)
                            bwd_ms = triton.testing.do_bench(lambda: out.backward(dO, retain_graph=True))
                        except Exception:
                            bwd_ms = float('nan')
                    else:
                        bwd_ms = float('nan')
                        
                    row_data[f"{name}_bwd"] = bwd_ms
                    row_data[f"{name}_e2e"] = fwd_ms + bwd_ms

                results.append(row_data)
                
                # Cleanup to prevent OOM
                del Q, K, V, dO
                if 'out' in locals(): del out
                torch.cuda.empty_cache()
                
                # Print progress row
                print(f"{seq_len:<8} {dim:<4} {row_data['dtype']:<8} | "
                      f"{row_data['Torch(Eager)_fwd']:<12.4f} {row_data['Torch(Compile)_fwd']:<14.4f} {row_data['FlashAttn_fwd']:<12.4f}")

    # Save detailed CSV and Plot results
    import datetime
    import os
    import importlib.util

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get GPU Info for folder naming
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0).replace(" ", "_").replace("/", "-")
        gpu_info = f"{gpu_count}x{gpu_name}"
    else:
        gpu_info = "CPU"
    
    # Base directory: cs336_systems/flash_attention
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # New timestamped sub-directory with GPU info
    output_dir = os.path.join(base_dir, f"benchmark_run_{gpu_info}_{timestamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    csv_filename = os.path.join(output_dir, "benchmark_results.csv")
    
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False)
    print(f"\nBenchmark complete. Results saved to {csv_filename}")
    print(df)
    
    # Run plotting script
    print("\nGenerating plots...")
    try:
        # Import plot_benchmark_results dynamically
        plot_script_path = os.path.join(base_dir, "plot_benchmark_results.py")
        spec = importlib.util.spec_from_file_location("plot_benchmark_results", plot_script_path)
        plot_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plot_module)
        
        # Call the plot function
        plot_module.plot_benchmark(csv_filename, output_dir)
        print(f"Plots generated in {output_dir}")
    except Exception as e:
        print(f"Failed to generate plots: {e}")

if __name__ == "__main__":
    run_benchmark()
