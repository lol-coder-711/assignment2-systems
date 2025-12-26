import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

class ToyLanguageModel(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=32, nhead=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Using a single Transformer layer for simplicity and speed
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dropout=0.0) # need to set dropout to 0.0 for deterministic results
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: (B, T)
        x = self.embedding(x)  # (B, T, D)
        x = self.transformer(x) # (B, T, D)
        logits = self.lm_head(x) # (B, T, V)
        return logits

def naive_ddp_all_reduce(model, world_size, return_stats=False, flatten=False):
    """
    Performs all-reduce on parameter gradients.
    
    Args:
        model: PyTorch model with gradients to reduce
        world_size: Number of processes in the distributed group
        return_stats: If True, return total bytes transferred
        flatten: If True, flatten all gradients into single tensor before all-reduce.
                 If False, all-reduce each parameter gradient individually (naive).
    
    Returns:
        If return_stats=True: total_bytes (int)
        If return_stats=False: None
    """
    total_bytes = 0
    
    if flatten:
        # Collect all gradients
        grads = [p.grad for p in model.parameters() 
                 if p.requires_grad and p.grad is not None]
        
        if len(grads) == 0:
            return 0 if return_stats else None
        
        # Flatten into single tensor
        flat_grads = torch._utils._flatten_dense_tensors(grads)
        
        # Single all-reduce call
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads /= world_size
        
        # Unflatten back to original shapes
        unflattened = torch._utils._unflatten_dense_tensors(flat_grads, grads)
        
        # In-place update each parameter's gradient
        # Using copy_() to preserve gradient tensor references (important for optimizer)
        for grad, new_grad in zip(grads, unflattened):
            grad.copy_(new_grad)
        
        if return_stats:
            total_bytes = flat_grads.numel() * flat_grads.element_size()
    else:
        # Original: all-reduce each parameter individually
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size
                if return_stats:
                    total_bytes += param.grad.numel() * param.grad.element_size()
    
    if return_stats:
        return total_bytes

def run_training(rank, world_size):
    # Setup process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Set seed for reproducibility
    torch.manual_seed(0)
    
    # 1. Initialize Single Process Model (Reference)
    vocab_size = 100
    model_single = ToyLanguageModel(vocab_size=vocab_size)
    optimizer_single = optim.SGD(model_single.parameters(), lr=0.01)

    # 2. Initialize DDP Model
    # Deepcopy to ensure identical start
    model_ddp = deepcopy(model_single)
    optimizer_ddp = optim.SGD(model_ddp.parameters(), lr=0.01)

    # 3. Generate Data (Simulating Token IDs)
    # Total dataset size: 20
    # Sequence length: 10
    torch.manual_seed(42)  # Ensure data is same across ranks (locally generated same data)
    X = torch.randint(0, vocab_size, (20, 10))
    Y = torch.randint(0, vocab_size, (20, 10))  # Flattened targets or keep same shape? CrossEntropy expects (B, C, T) or (N, C)

    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for step in range(5):
        # --- Single Process Step ---
        optimizer_single.zero_grad()
        output_single = model_single(X) # (B, T, V)
        # Reshape for CrossEntropyLoss: (B*T, V) vs (B*T)
        loss_single = criterion(output_single.view(-1, vocab_size), Y.view(-1))
        loss_single.backward()
        optimizer_single.step()

        # --- DDP Step ---
        # Slice data for this rank
        batch_size_per_rank = len(X) // world_size
        start_idx = rank * batch_size_per_rank
        end_idx = start_idx + batch_size_per_rank
        
        x_local = X[start_idx:end_idx]
        y_local = Y[start_idx:end_idx]

        optimizer_ddp.zero_grad()
        output_ddp = model_ddp(x_local) # (local_bs, T, V)
        loss_ddp = criterion(output_ddp.view(-1, vocab_size), y_local.view(-1))
        loss_ddp.backward()

        # Naive All-Reduce
        naive_ddp_all_reduce(model_ddp, world_size)
        
        optimizer_ddp.step()

        # --- Verification ---
        # Check if weights match on Rank 0
        if rank == 0:
            with torch.no_grad():
                for p_single, p_ddp in zip(model_single.parameters(), model_ddp.parameters()):
                    if not torch.allclose(p_single, p_ddp, atol=1e-6):
                        print(f"Step {step}: Mismatch found!")
                        # print(f"Single: {p_single}")
                        # print(f"DDP: {p_ddp}")
                        assert False, "Weights mismatch between single process and DDP!"

    if rank == 0:
        print("DDP implementation verified successfully!")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(run_training, args=(world_size,), nprocs=world_size, join=True)
