import torch
import torch.nn as nn
import timeit
import argparse
import sys
import os
import statistics
import contextlib

# Ensure we can import from cs336-basics
# Go up one level since we are in cs336_systems/
sys.path.append(os.path.join(os.path.dirname(__file__), "../cs336-basics"))

from cs336_basics.model import BasicsTransformerLM
import cs336_basics.model
import math
import torch.cuda.nvtx as nvtx
from einops import einsum
from cs336_basics.nn_utils import softmax

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range("final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
        
    return output

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def get_args():
    parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM forward and backward passes.")
    
    parser.add_argument("--config", type=str, choices=CONFIGS.keys(), help="Predefined model configuration")
    
    # Hyperparameters (defaults will be used if config not specified)
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=128, help="Context length")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension") 
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1024, help="Feed-forward dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta")
    
    # Benchmarking config
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--warmup_steps", "-w", type=int, default=5, help="Number of warm-up steps")
    parser.add_argument("--steps", "-n", type=int, default=10, help="Number of steps to time")
    parser.add_argument("--mode", choices=["forward", "backward", "training"], default="forward", help="Benchmarking mode: forward only, forward+backward, or full training step (fwd+bwd+opt)")
    parser.add_argument("--mixed_precision", choices=["no", "bf16"], default="no", help="Use mixed precision (bf16) or not")
    parser.add_argument("--enable_memory_profiling", action="store_true", help="Enable memory profiling")
    parser.add_argument("--memory_snapshot_file", type=str, default="profiling_result/memory_snapshot.pickle", help="Output file for memory snapshot")
    
    args = parser.parse_args()
    
    # Apply config if specified
    if args.config:
        config = CONFIGS[args.config]
        for k, v in config.items():
            setattr(args, k, v)
            
    return args

def run_benchmark():
    args = get_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Using device: {device}")
    print(f"Configuration: {args.config if args.config else 'Custom'}")
    print(f"Model params: d_model={args.d_model}, layers={args.num_layers}, heads={args.num_heads}, d_ff={args.d_ff}")
    
    # 1. Initialize model
    print("Initializing model...", end=" ", flush=True)
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)
    print("Done.")
    
    # 2. Generate random batch of data
    print("Generating data...")
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    
    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    # Lists to store timings
    forward_times = []
    backward_times = []
    
    # Helper for a single step
    def step_fn(measure=False):
        # Determine mixed precision context
        if args.mixed_precision == "bf16":
            mp_context = torch.autocast(device_type=device, dtype=torch.bfloat16)
        else:
            mp_context = contextlib.nullcontext()

        # Forward
        t0 = timeit.default_timer()
        with mp_context:
            logits = model(x)
        sync()
        t1 = timeit.default_timer()
        
        if measure:
            forward_times.append(t1 - t0)
        
        if args.mode == "backward":
            with mp_context:
                loss = criterion(logits.view(-1, args.vocab_size), y.view(-1))
            
            # Backward
            t2 = timeit.default_timer()
            loss.backward()
            model.zero_grad(set_to_none=True)
            sync()
            t3 = timeit.default_timer()
            
            if measure:
                backward_times.append(t3 - t2)

        elif args.mode == "training":
            with mp_context:
                loss = criterion(logits.view(-1, args.vocab_size), y.view(-1))
            
            # Backward
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            sync()
            t3 = timeit.default_timer()
            
            if measure:
                backward_times.append(t3 - t1) # For training, track time after forward pass (Backward + Opt)

    # 3. Warm-up
    print(f"Running {args.warmup_steps} warm-up steps...")
    for _ in range(args.warmup_steps):
        step_fn(measure=False)

    if args.enable_memory_profiling:
        print("Starting memory recording...")
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        
    # 4. Timing
    print(f"Running {args.steps} steps for timing...")
    for _ in range(args.steps):
        step_fn(measure=True)

    if args.enable_memory_profiling:
        print(f"Saving memory snapshot to {args.memory_snapshot_file}...")
        torch.cuda.memory._dump_snapshot(args.memory_snapshot_file)
        torch.cuda.memory._record_memory_history(enabled=None)
    
    # 5. Report results
    print(f"\nResults (over {args.steps} steps):")
    
    f_mean = statistics.mean(forward_times)
    f_stdev = statistics.stdev(forward_times) if len(forward_times) > 1 else 0.0
    print(f"Forward pass:  {f_mean:.4f} s ± {f_stdev:.4f} s")
    
    if args.mode == "backward":
        b_mean = statistics.mean(backward_times)
        b_stdev = statistics.stdev(backward_times) if len(backward_times) > 1 else 0.0
        print(f"Backward pass: {b_mean:.4f} s ± {b_stdev:.4f} s")
        print(f"Total step:    {(f_mean + b_mean):.4f} s")
    
    if args.mode == "training":
        # In training mode, backward_times measures the duration of (Backward Pass + Optimizer Step)
        b_mean = statistics.mean(backward_times)
        b_stdev = statistics.stdev(backward_times) if len(backward_times) > 1 else 0.0
        print(f"Backward + Optimizer: {b_mean:.4f} s ± {b_stdev:.4f} s")
        print(f"Total Step Time:      {(f_mean + b_mean):.4f} s")

if __name__ == "__main__":
    run_benchmark()
