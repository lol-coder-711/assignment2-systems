import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import logging

from cs336_systems.dist_training.ddp_naive import naive_ddp_all_reduce

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



def run_benchmark(rank, world_size):
    # 1. Setup Environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356' # Use different port than naive test
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ["GLOO_LOG_LEVEL"] = "ERROR"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    os.environ["GLOG_minloglevel"] = "2" # Suppress Glog Warnings (0=INFO, 1=WARNING, 2=ERROR)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        backend = "nccl"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        backend = "gloo"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    if rank == 0:
        logger.info(f"Running on {device} with backend {backend}")

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
    # Note: BasicsTransformerLM expects explicit args, unpacking config
    try:
        model = BasicsTransformerLM(**config).to(device)
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
    num_warmup = 1
    num_steps = 3
    
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
        loss.backward()

        # Communication (Naive All-Reduce)
        if not is_warmup:
            comm_timer.start()
            
        step_bytes = naive_ddp_all_reduce(model, world_size, return_stats=True)
        
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
        
        print("\n" + "="*40)
        print(f"Benchmark Results (Device: {device})")
        print(f"Model Config: {config}")
        print(f"World Size: {world_size}")
        print(f"Average Step Time: {avg_step_time:.2f} ms")
        print(f"Average Comm Time: {avg_comm_time:.2f} ms")
        print(f"Average All-Reduce Data: {avg_step_bytes/1e6:.2f} MB")
        print(f"Communication Overhead: {comm_ratio:.2f}%")
        print("="*40 + "\n")

    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    # Standard local test world size
    world_size = 4
    mp.spawn(run_benchmark, args=(world_size,), nprocs=world_size, join=True)
