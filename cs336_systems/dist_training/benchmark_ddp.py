import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import logging

from cs336_systems.dist_training.ddp_naive import naive_ddp_all_reduce
from cs336_systems.dist_training.ddp_individual_parameters import DDP

try:
    from cs336_basics.model import BasicsTransformerLM
except ImportError:
    # Fallback/Mock for standalone testing if cs336_basics isn't in path
    print("Warning: cs336_basics not found, using simple Mock model")
    class BasicsTransformerLM(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.linear = nn.Linear(10, 10)
        def forward(self, x):
            return self.linear(x.float())

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Timer:
    def __init__(self, device):
        self.device = device
        self.start_t = None
        self.end_t = None
        if self.device.type == 'cuda':
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.device.type == 'cuda':
            self.start_event.record()
        else:
            self.start_t = time.time()

    def stop(self):
        if self.device.type == 'cuda':
            self.end_event.record()
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event) # ms
        elif self.device.type == 'mps':
            torch.mps.synchronize()
            self.end_t = time.time()
            return (self.end_t - self.start_t) * 1000 # ms
        else:
            self.end_t = time.time()
            return (self.end_t - self.start_t) * 1000 # ms



def run_benchmark(rank, world_size, flatten=False, overlap=False, results_queue=None):
    """
    Run benchmark for DDP training.
    
    Args:
        rank: Process rank
        world_size: Number of processes
        flatten: Whether to use flattened all-reduce (ignored if overlap=True)
        overlap: Whether to use DDP with async gradient hooks for overlap
        results_queue: Multiprocessing queue to store results
    """

        # 1. Setup Environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356' # Use different port than naive test
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ["GLOO_LOG_LEVEL"] = "ERROR"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    os.environ["GLOG_minloglevel"] = "2" # Suppress Glog Warnings

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        backend = "nccl"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        backend = "gloo"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    if rank == 0:
        if overlap:
            method_name = "Overlap"
        elif flatten:
            method_name = "Flattened"
        else:
            method_name = "Naive"
        logger.info(f"Running on {device} with backend {backend}, Method: {method_name}")

    # 2. Model Configuration
    if device.type == "cuda":
        # L Config (GPT-2 Large equivalent)
        config = {
            "vocab_size": 50257,
            "context_length": 1024,
            "d_model": 1280,
            "num_layers": 36,
            "num_heads": 20,
            "d_ff": 5120, # 4 * 1280
            "rope_theta": 10000.0,
        }
    else:
        # Small Config for CPU/Debugging/MPS (Memory constraint)
        config = {
            "vocab_size": 10000,
            "context_length": 128,
            "d_model": 256,
            "num_layers": 16,
            "num_heads": 4,
            "d_ff": 1024,
            "rope_theta": 10000.0,
        }

    if rank == 0:
        logger.info(f"Model keys: {config.keys()}")

    # Initialize Model
    try:
        model = BasicsTransformerLM(**config).to(device)
        
        # Wrap with DDP if overlap mode
        if overlap:
            model = DDP(model)
    except Exception as e:
        logger.error(f"Failed to init model: {e}")
        return

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 3. Data Generation (Random)
    batch_size = 1 # Per GPU/Rank batch size
    seq_len = config["context_length"]
    vocab_size = config["vocab_size"]
    
    # Pre-generate data to avoid overhead during timing
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    # 4. Benchmarking Loop
    num_warmup = 5
    num_steps = 10
    
    step_timer = Timer(device)
    comm_timer = Timer(device)
    
    total_step_time = 0.0
    total_comm_time = 0.0
    step_bytes = 0

    model.train()

    for step in range(num_warmup + num_steps):
        is_warmup = step < num_warmup
        
        # Start Step Timing
        if not is_warmup:
            step_timer.start()

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        
        # Time backward separately
        if not is_warmup:
            backward_timer = Timer(device)
            backward_timer.start()
        
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push("Backward")
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

        if not is_warmup:
            backward_ms = backward_timer.stop()
            if rank == 0 and overlap:
                logger.info(f"  Backward: {backward_ms:.2f}ms")

        # Communication
        if overlap:
            # Overlap mode: hooks already triggered async all-reduce during backward
            if not is_warmup:
                comm_timer.start()
            
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push("Wait for Sync")
            model.finish_gradient_synchronization()
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
                
            if not is_warmup:
                step_comm_ms = comm_timer.stop()
                total_comm_time += step_comm_ms
            # For overlap, we can't easily measure bytes, use approximate
            if step == num_warmup:
                # Calculate bytes once (all params)
                step_bytes = sum(p.grad.numel() * p.grad.element_size() 
                                for p in model.module.parameters() 
                                if p.requires_grad and p.grad is not None)
        else:
            # Naive/Flattened: blocking all-reduce
            if not is_warmup:
                comm_timer.start()
            
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push("Communication")
            step_bytes = naive_ddp_all_reduce(model, world_size, return_stats=True, flatten=flatten)
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
            
            if not is_warmup:
                step_comm_ms = comm_timer.stop()
                total_comm_time += step_comm_ms

        optimizer.step()

        # Stop Step Timing
        if not is_warmup:
            step_total_ms = step_timer.stop()
            total_step_time += step_total_ms
            
            if rank == 0:
                logger.info(f"Step {step}: Total={step_total_ms:.2f}ms, Comm={step_comm_ms:.2f}ms, Data={step_bytes/1e6:.2f}MB")

    # 5. Reporting
    if rank == 0:
        avg_step_time = total_step_time / num_steps
        avg_comm_time = total_comm_time / num_steps
        comm_ratio = (avg_comm_time / avg_step_time) * 100 if avg_step_time > 0 else 0
        avg_step_bytes = step_bytes
        
        result = {
            'world_size': world_size,
            'flatten': flatten,
            'overlap': overlap,
            'device': str(device),
            'avg_step_time': avg_step_time,
            'avg_comm_time': avg_comm_time,
            'avg_step_bytes': avg_step_bytes,
            'comm_ratio': comm_ratio,
        }
        
        if results_queue is not None:
            results_queue.put(result)

    dist.destroy_process_group()

def print_results_table(results):
    """
    Print benchmark results in a nicely formatted table.
    """
    print("\n" + "="*100)
    print("DDP BENCHMARK RESULTS: Naive vs Flattened vs Overlap")
    print("="*100)
    
    # Header
    print(f"{'Method':<12} {'World Size':<12} {'Avg Step (ms)':<16} {'Avg Comm (ms)':<16} {'Data (MB)':<12} {'Comm %':<10}")
    print("-" * 100)
    
    # Sort results by world_size, then by method (naive, flattened, overlap)
    def method_key(r):
        if r.get('overlap'):
            return 2  # Overlap
        elif r.get('flatten'):
            return 1  # Flattened
        else:
            return 0  # Naive
    
    results = sorted(results, key=lambda x: (x['world_size'], method_key(x)))
    
    # Print rows
    for r in results:
        if r.get('overlap'):
            method = "Overlap"
        elif r.get('flatten'):
            method = "Flattened"
        else:
            method = "Naive"
        print(f"{method:<12} {r['world_size']:<12} {r['avg_step_time']:<16.2f} {r['avg_comm_time']:<16.2f} "
              f"{r['avg_step_bytes']/1e6:<12.2f} {r['comm_ratio']:<10.2f}")
    
    print("="*100 + "\n")
    
    # Print speedup analysis
    print("SPEEDUP ANALYSIS (vs Naive baseline):")
    print("-" * 80)
    world_sizes = sorted(set(r['world_size'] for r in results))
    for ws in world_sizes:
        naive = next((r for r in results if r['world_size'] == ws and not r.get('flatten') and not r.get('overlap')), None)
        flattened = next((r for r in results if r['world_size'] == ws and r.get('flatten')), None)
        overlap = next((r for r in results if r['world_size'] == ws and r.get('overlap')), None)
        
        if naive:
            print(f"\nWorld Size {ws}:")
            if flattened:
                step_speedup = naive['avg_step_time'] / flattened['avg_step_time']
                comm_speedup = naive['avg_comm_time'] / flattened['avg_comm_time']
                print(f"  Flattened - Step: {step_speedup:.2f}x, Comm: {comm_speedup:.2f}x")
            if overlap:
                step_speedup = naive['avg_step_time'] / overlap['avg_step_time']
                comm_speedup = naive['avg_comm_time'] / overlap['avg_comm_time']
                print(f"  Overlap   - Step: {step_speedup:.2f}x, Comm: {comm_speedup:.2f}x")
    print("-" * 80 + "\n")


if __name__ == "__main__":
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Test configurations
    # Modified to run only the required 1 node 2 GPUs cases
    world_sizes = [2] if torch.cuda.is_available() else [2, 4, 8]
    # Two methods: Naive and Overlap
    test_configs = [
        {'flatten': False, 'overlap': False},  # Naive
        {'flatten': True, 'overlap': False},   # Flattened  
        {'flatten': False, 'overlap': True},   # Overlap
    ]
    
    all_results = []
    
    for world_size in world_sizes:
        for config in test_configs:
            flatten = config['flatten']
            overlap = config['overlap']
            
            if overlap:
                method_name = "Overlap"
            elif flatten:
                method_name = "Flattened"
            else:
                method_name = "Naive"
                
            print(f"\n{'='*60}")
            print(f"Running: Method={method_name}, World Size={world_size}")
            print(f"{'='*60}\n")
            
            # Create a queue to collect results from rank 0
            ctx = mp.get_context('spawn')
            results_queue = ctx.Queue()
            
            # Run benchmark
            mp.spawn(
                run_benchmark,
                args=(world_size, flatten, overlap, results_queue),
                nprocs=world_size,
                join=True
            )
            
            # Collect result
            if not results_queue.empty():
                result = results_queue.get()
                all_results.append(result)
    
    # Print final comparison table
    if all_results:
        print_results_table(all_results)
