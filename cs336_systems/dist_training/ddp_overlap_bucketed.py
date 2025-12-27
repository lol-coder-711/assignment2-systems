import torch
import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    """
    Distributed Data Parallel wrapper that overlaps gradient communication
    with backward pass computation using gradient bucketing.
    
    Buckets gradients of similar size together to reduce the number of
    communication calls while maintaining overlap with computation.
    """
    
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float = 25.0):
        """
        Wrap a PyTorch module for distributed training with gradient bucketing.
        
        Args:
            module: PyTorch model to wrap
            bucket_size_mb: Maximum size of each gradient bucket in megabytes.
                           Each bucket will hold at most bucket_size_mb of parameters.
        """
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        
        # Get distributed training info
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # List to store communication handles for async all-reduce operations
        self._handles = []
        
        # Broadcast initial parameters from rank 0 to all other ranks
        # This ensures all ranks start with the same model weights
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        # Dynamic bucketing state (runtime)
        self._current_bucket_params = []  # List of param for current bucket
        self._current_bucket_grads = []   # List of gradients in current bucket
        self._acc_bytes = 0
        
        def _post_accumulate_grad_hook(param):
            """Hook that dynamically buckets gradients"""
            # Ensure grad exists and is not None
            if param.grad is None:
                return

            param_bytes = param.numel() * param.grad.element_size()
            
            # Check if adding this param would overflow the bucket
            # Also check if _current_bucket_grads is not empty to avoid creating an empty bucket
            if self._acc_bytes + param_bytes > self.bucket_size_mb * 1024 * 1024 and self._current_bucket_grads:
                # Current bucket is full, all-reduce it!
                flat_grads = torch._utils._flatten_dense_tensors(self._current_bucket_grads)
                handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
                # Store: (handle, flattened_grad, list_of_(param, original_shape))
                self._handles.append((handle, flat_grads, self._current_bucket_params))
                
                # Start new bucket with current param
                self._current_bucket_params = [param]
                self._current_bucket_grads = [param.grad]
                self._acc_bytes = param_bytes
            else:
                # Add to current bucket
                self._current_bucket_params.append(param)
                self._current_bucket_grads.append(param.grad)
                self._acc_bytes += param_bytes
        
        # Register hooks for all parameters (reverse order for better overlap)
        for param in reversed(list(self.module.parameters())):
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(_post_accumulate_grad_hook)
    
    def forward(self, *inputs, **kwargs):
        """
        Call wrapped module's forward method.
        
        Args:
            *inputs: Positional arguments to forward
            **kwargs: Keyword arguments to forward
            
        Returns:
            Output of the wrapped module
        """
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        """
        Wait for all asynchronous all-reduce operations to complete.
        Must be called after backward() and before optimizer.step().
        
        This method:
        1. Handles the last incomplete bucket
        2. Waits for all pending all-reduce operations to complete
        3. Unflattens and averages the gradients
        4. Clears state for next iteration
        """
        # Handle the last bucket if any params remain
        if self._current_bucket_grads:
            flat_grads = torch._utils._flatten_dense_tensors(self._current_bucket_grads)
            handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
            self._handles.append((handle, flat_grads, list(self._current_bucket_params)))
        
        # Wait for all all-reduce operations and unflatten
        for handle, flat_grads, bucket_params in self._handles:
            # Wait for this all-reduce to complete
            handle.wait()
            
            # Average the gradient (all_reduce gave us the sum)
            flat_grads.div_(self.world_size)
            
            # Unflatten and write back to param.grad
            # The `bucket_params` contains param
            # We need the actual grad tensors for unflattening to get their original shapes
            original_grad_tensors = [p.grad for p in bucket_params]
            unflattened = torch._utils._unflatten_dense_tensors(flat_grads, original_grad_tensors)
            for param, grad in zip(bucket_params, unflattened):
                param.grad.copy_(grad)
        
        # Clear state for next iteration
        self._handles.clear()
        self._current_bucket_params.clear()
        self._current_bucket_grads.clear()
        self._acc_bytes = 0
