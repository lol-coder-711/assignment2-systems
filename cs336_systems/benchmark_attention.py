import torch
import timeit
import sys
import os
import math
from einops import einsum

# Add parent directory to path to import cs336_basics
sys.path.append(os.path.join(os.path.dirname(__file__), "../cs336-basics"))
from cs336_basics.nn_utils import softmax

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Re-implementation or import of the naive attention
    d_k = K.shape[-1]
    # Naive implementation: materializes (Batch, Seq, Seq) matrix
    attention_scores = einsum(Q, K, "batch query d_k, batch key d_k -> batch query key") / math.sqrt(d_k)
    
    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    
    attention_weights = softmax(attention_scores, dim=-1)
    return einsum(attention_weights, V, "batch query key, batch key d_v -> batch query d_v")

def benchmark():
    BATCH_SIZE = 8
    D_HEADS = [16, 32, 64, 128]
    SEQ_LENS = [256, 1024, 4096, 8192, 16384]
    
    # 20GB limit awareness: 16k * 16k * 4 bytes * 8 batch = 8GB just for the matrix (in FP32).
    # Gradients will double/triple this.
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on {device}")
    
    compiled_attn = torch.compile(scaled_dot_product_attention)
    
    print(f"{'d_head':<8} | {'seq_len':<8} | {'Van Fwd':<10} | {'Van Bwd':<10} | {'Cmp Fwd':<10} | {'Cmp Bwd':<10} | {'Mem (GB)':<10} | {'Status'}")
    print("-" * 100)

    for d_head in D_HEADS:
        for seq_len in SEQ_LENS:
            try:
                # Setup
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                Q = torch.randn(BATCH_SIZE, seq_len, d_head, device=device, requires_grad=True)
                K = torch.randn(BATCH_SIZE, seq_len, d_head, device=device, requires_grad=True)
                V = torch.randn(BATCH_SIZE, seq_len, d_head, device=device, requires_grad=True)
                
                # --- VANILLA BENCHMARK ---
                # Warmup
                for _ in range(5):
                    out = scaled_dot_product_attention(Q, K, V)
                    loss = out.sum()
                    loss.backward()
                    Q.grad = None; K.grad = None; V.grad = None
                torch.cuda.synchronize()
                
                # Measure Forward
                t0 = timeit.default_timer()
                for _ in range(100):
                    out = scaled_dot_product_attention(Q, K, V)
                torch.cuda.synchronize()
                t1 = timeit.default_timer()
                van_fwd_ms = (t1 - t0) * 1000 / 100 
                
                # Measure Memory before Backward (Vanilla)
                torch.cuda.reset_peak_memory_stats()
                out = scaled_dot_product_attention(Q, K, V)
                mem_bytes = torch.cuda.max_memory_allocated()
                loss = out.sum()
                
                # Measure Backward
                bwd_times = []
                for _ in range(100):
                    out_loop = scaled_dot_product_attention(Q, K, V)
                    loss_loop = out_loop.sum()
                    torch.cuda.synchronize()
                    t_b_start = timeit.default_timer()
                    loss_loop.backward()
                    torch.cuda.synchronize()
                    t_b_end = timeit.default_timer()
                    bwd_times.append(t_b_end - t_b_start)
                    Q.grad = None; K.grad = None; V.grad = None
                van_bwd_ms = (sum(bwd_times) / len(bwd_times)) * 1000
                
                # Cleanup Vanilla variables before Compiled run to save memory
                del out, loss, out_loop, loss_loop
                Q.grad = None; K.grad = None; V.grad = None
                torch.cuda.empty_cache()
                import gc; gc.collect()

                # --- COMPILED BENCHMARK ---
                # Re-create fresh graph inputs just in case (though resetting grads is enough)
                
                # Warmup (Important: This triggers compilation!)
                for _ in range(5):
                    out = compiled_attn(Q, K, V)
                    loss = out.sum()
                    loss.backward()
                    Q.grad = None; K.grad = None; V.grad = None
                torch.cuda.synchronize()
                
                # Measure Compiled Forward
                t0 = timeit.default_timer()
                for _ in range(100):
                    out = compiled_attn(Q, K, V)
                torch.cuda.synchronize()
                t1 = timeit.default_timer()
                cmp_fwd_ms = (t1 - t0) * 1000 / 100
                
                # Measure Compiled Backward
                # Note: compiled objects usually manage their own backward graph compilation
                bwd_times_cmp = []
                for _ in range(100):
                    out_loop = compiled_attn(Q, K, V)
                    loss_loop = out_loop.sum()
                    torch.cuda.synchronize()
                    t_b_start = timeit.default_timer()
                    loss_loop.backward()
                    torch.cuda.synchronize()
                    t_b_end = timeit.default_timer()
                    bwd_times_cmp.append(t_b_end - t_b_start)
                    Q.grad = None; K.grad = None; V.grad = None
                cmp_bwd_ms = (sum(bwd_times_cmp) / len(bwd_times_cmp)) * 1000

                # Check if compiled version memory usage is significantly different? 
                # Prompt doesn't mandate tracking second memory stat, but we can stick to vanilla memory for the table column
                # or we could update it. For now, using Vanilla memory as reference for OOM check.

                print(f"{d_head:<8} | {seq_len:<8} | {van_fwd_ms:<10.4f} | {van_bwd_ms:<10.4f} | {cmp_fwd_ms:<10.4f} | {cmp_bwd_ms:<10.4f} | {mem_bytes/1e9:<10.4f} | OK")
                
                # Cleanup
                if 'out' in locals(): del out
                if 'loss' in locals(): del loss
                if 'out_loop' in locals(): del out_loop
                if 'loss_loop' in locals(): del loss_loop
                del Q, K, V
                torch.cuda.empty_cache()
                import gc; gc.collect()

            except torch.cuda.OutOfMemoryError:
                print(f"{d_head:<8} | {seq_len:<8} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'OOM':<10} | OOM")
                # Attempt to cleanup
                if 'Q' in locals(): del Q
                if 'K' in locals(): del K
                if 'V' in locals(): del V
                if 'out' in locals(): del out
                if 'loss' in locals(): del loss
                if 'out_loop' in locals(): del out_loop
                if 'loss_loop' in locals(): del loss_loop
                torch.cuda.empty_cache()
                import gc; gc.collect()

if __name__ == "__main__":
    benchmark()
