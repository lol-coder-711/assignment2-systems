import os
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime, timedelta
import platform

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    # Suppress Gloo warning on Mac by binding to loopback if using Gloo
    if backend == "gloo" and platform.system() == "Darwin":
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0"

    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timedelta(minutes=5))

def cleanup():
    dist.destroy_process_group()

def benchmark_worker(rank, world_size, backend, device, data_sizes_mb, num_iters, warmup_iters, result_queue):
    """
    The worker function that runs in each process.
    """
    try:
        setup(rank, world_size, backend)
        
        results = []
        
        for size_mb in data_sizes_mb:
            # Calculate number of elements (assuming float32 = 4 bytes)
            num_elements = int(size_mb * 1024 * 1024 / 4)
            
            if device == 'cuda':
                # Set device for this process (assuming 1 GPU per process for simplicity in this benchmark)
                # For single-node multi-GPU, we typically map rank to device_id.
                # However, for the provided Mac/CPU case, this branch won't execute.
                # If running on a GPU node with fewer GPUs than ranks, this might oversubscribe,
                # but standard practice is rank % num_gpus.
                if torch.cuda.is_available():
                    device_id = rank % torch.cuda.device_count()
                    torch.cuda.set_device(device_id)
                    tensor = torch.randn(num_elements, device=device_id)
                else:
                    raise RuntimeError("CUDA specified but not available.")
            else:
                tensor = torch.randn(num_elements)
            
            # Warmup
            for _ in range(warmup_iters):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
            
            # Synchronize before timing (especially critical for CUDA)
            if device == 'cuda':
                torch.cuda.synchronize()
            dist.barrier() # Ensure all processes are ready to start
            
            start_time = time.time()
            for _ in range(num_iters):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
            
            # Synchronize after timing
            if device == 'cuda':
                torch.cuda.synchronize()
            dist.barrier() # Ensure all processes are finished
            
            end_time = time.time()
            
            avg_latency_ms = ((end_time - start_time) / num_iters) * 1000
            
            if rank == 0:
                print(f"Rank {rank}: Size {size_mb}MB, World Size {world_size} -> {avg_latency_ms:.4f} ms")
                results.append({
                    "world_size": world_size,
                    "data_size_mb": size_mb,
                    "latency_ms": avg_latency_ms,
                    "backend": backend,
                    "device": device
                })
        
        # Only rank 0 pushes results to queue to avoid duplicates
        if rank == 0:
            result_queue.put(results)
            
    except Exception as e:
        print(f"Rank {rank} encountered an error: {e}")
        raise e
    finally:
        cleanup()

def run_benchmark(backend, device, world_size, distinct_sizes, num_iters=20, warmup_iters=5):
    print(f"\n--- Benchmarking World Size: {world_size} | Backend: {backend} | Device: {device} ---")
    
    # Use multiprocessing Queue to get results back from the spawned process
    # Note: mp.Queue works with mp.spawn usually, but passing it as an arg is key.
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    
    mp.spawn(
        fn=benchmark_worker,
        args=(world_size, backend, device, distinct_sizes, num_iters, warmup_iters, result_queue),
        nprocs=world_size,
        join=True
    )
    
    # Retrieve results from queue
    # Since only rank 0 puts, we expect 1 item (which is a list of results)
    run_results = []
    while not result_queue.empty():
        run_results.extend(result_queue.get())
        
    return run_results

def plot_results(csv_file, output_dir):
    if not os.path.exists(csv_file):
        print("No CSV file found to plot.")
        return

    df = pd.read_csv(csv_file)
    
    plt.figure(figsize=(10, 6))
    
    # Group by World Size and plot
    world_sizes = sorted(df['world_size'].unique())
    
    for ws in world_sizes:
        subset = df[df['world_size'] == ws]
        subset = subset.sort_values('data_size_mb')
        plt.plot(subset['data_size_mb'], subset['latency_ms'], marker='o', label=f'World Size {ws}')
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Data Size (MB)")
    plt.ylabel("Latency (ms)")
    plt.title(f"All-Reduce Latency ({df['backend'].iloc[0]}/{df['device'].iloc[0]})")
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    
    output_file = os.path.join(output_dir, "all_reduce_plot.png")
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark distributed all_reduce")
    parser.add_argument("--backend", type=str, default=None, choices=['gloo', 'nccl'], help="Communication backend")
    parser.add_argument("--device", type=str, default=None, choices=['cpu', 'cuda'], help="Device type")
    parser.add_argument("--world_sizes", type=str, default="2,4,6", help="Comma-separated list of world sizes")
    parser.add_argument("--data_sizes_mb", type=str, default="1,10,100,1000", help="Comma-separated list of data sizes in MB")
    parser.add_argument("--num_iters", type=int, default=10, help="Number of timing iterations")
    parser.add_argument("--warmup_iters", type=int, default=5, help="Number of warmup iterations")
    args = parser.parse_args()

    # Auto-detect defaults
    if args.backend is None:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            args.backend = 'nccl'
        else:
            args.backend = 'gloo'
        print(f"Auto-detected backend: {args.backend}")

    if args.device is None:
        if args.backend == 'nccl':
            args.device = 'cuda'
        else:
            args.device = 'cpu'
        print(f"Auto-detected device: {args.device}")

    # Parse lists
    world_sizes = [int(x) for x in args.world_sizes.split(',')]
    data_sizes_mb = [float(x) for x in args.data_sizes_mb.split(',')]

    all_results = []

    for ws in world_sizes:
        # Check if we have enough GPUs if running on CUDA
        if args.device == 'cuda' and torch.cuda.device_count() < ws:
             print(f"Warning: Requesting {ws} processes but only {torch.cuda.device_count()} GPUs available. Processes will share GPUs.")
        
        # On Mac/CPU, spawning too many processes might be heavy, but usually fine for 2-6.
        # However, mp.spawn limits depending on start method.
        
        try:
            results = run_benchmark(
                backend=args.backend,
                device=args.device,
                world_size=ws,
                distinct_sizes=data_sizes_mb,
                num_iters=args.num_iters,
                warmup_iters=args.warmup_iters
            )
            all_results.extend(results)
        except Exception as e:
            print(f"Failed to run benchmark for world_size={ws}: {e}")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Determine basic path relative to script or cwd
    # Let's put it in cs336_systems/dist_training/benchmark_runs/...
    # Or just current directory if user prefers. 
    # User asked for "dist_training/下的sub repo里"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, f"benchmark_runs_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        csv_filename = os.path.join(output_dir, "all_reduce_results.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")
        
        # Plot
        try:
            plot_results(csv_filename, output_dir)
        except Exception as e:
            print(f"Plotting failed: {e}")
    else:
        print("No results collected.")

if __name__ == "__main__":
    main()
