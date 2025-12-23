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
    
    print(f"{'d_head':<8} | {'seq_len':<8} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Mem (GB)':<10} | {'Status'}")
    print("-" * 70)

    for d_head in D_HEADS:
        for seq_len in SEQ_LENS:
            try:
                # Setup
                # Shape: (Batch, Seq, D_Head)
                # Note: "don't use multihead attention (i.e. remove the head dimension)" 
                # effectively means treating input as (Batch, Seq, Feature) where Feature = d_head.
                
                # Reset memory
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                Q = torch.randn(BATCH_SIZE, seq_len, d_head, device=device, requires_grad=True)
                K = torch.randn(BATCH_SIZE, seq_len, d_head, device=device, requires_grad=True)
                V = torch.randn(BATCH_SIZE, seq_len, d_head, device=device, requires_grad=True)
                
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
                fwd_ms = (t1 - t0) * 1000 / 100 
                
                # Measure Memory before Backward
                # We need to do one forward pass and KEEP the graph alive
                # to measure activation memory
                torch.cuda.reset_peak_memory_stats()
                out = scaled_dot_product_attention(Q, K, V)
                mem_bytes = torch.cuda.max_memory_allocated()
                loss = out.sum()
                
                # Measure Backward
                t2 = timeit.default_timer()
                for _ in range(100):
                    # We need to re-run forward inside loop to have a graph to backprop through?
                    # Or usually benchmark usually does: forward + backward in loop?
                    # The prompt says "time 100 backward passes".
                    # Typically this implies: Fwd, Bwd, Fwd, Bwd... 
                    # If we just call loss.backward() 100 times on same graph, it's error (buffers freed).
                    # So we must loop full step.
                    
                    # Wait, prompt says: "Time 100 forward passes... Time 100 backward passes".
                    # This is slightly ambiguous. You usually can't time JUST backward 100 times in isolation efficiently without re-forwarding.
                    # UNLESS use retain_graph=True? But that accumulates memory.
                    # Standard practice: Time (Forward + Backward) - Time(Forward).
                    # Or just: Run Forward, Time Backward specific call.
                    
                    # I will do: Loop 100 times: (Forward, Time Backward Start, Backward, Time Backward End, Zero Grad)
                    pass
                
                # Correct approach for measuring Backward Time:
                bwd_times = []
                for _ in range(100):
                    # Re-create graph
                    out_loop = scaled_dot_product_attention(Q, K, V)
                    loss_loop = out_loop.sum()
                    
                    torch.cuda.synchronize()
                    t_b_start = timeit.default_timer()
                    loss_loop.backward()
                    torch.cuda.synchronize()
                    t_b_end = timeit.default_timer()
                    bwd_times.append(t_b_end - t_b_start)
                    
                    # Zero grads
                    Q.grad = None; K.grad = None; V.grad = None

                bwd_ms = (sum(bwd_times) / len(bwd_times)) * 1000
                
                # Cleanup
                if 'out' in locals(): del out
                if 'loss' in locals(): del loss
                if 'out_loop' in locals(): del out_loop
                if 'loss_loop' in locals(): del loss_loop
                del Q, K, V
                torch.cuda.empty_cache()
                import gc; gc.collect()

                print(f"{d_head:<8} | {seq_len:<8} | {fwd_ms:<10.4f} | {bwd_ms:<10.4f} | {mem_bytes/1e9:<10.4f} | OK")
                
            except torch.cuda.OutOfMemoryError:
                print(f"{d_head:<8} | {seq_len:<8} | {'N/A':<10} | {'N/A':<10} | {'OOM':<10} | OOM")
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
