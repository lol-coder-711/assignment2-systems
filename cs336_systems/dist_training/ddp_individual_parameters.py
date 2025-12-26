import torch
import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    """
    Distributed Data Parallel wrapper that overlaps gradient communication
    with backward pass computation using asynchronous all-reduce.
    """
    
    def __init__(self, module: torch.nn.Module):
        """
        Wrap a PyTorch module for distributed training with overlap.
        
        - Broadcasts initial weights to all ranks
        - Registers gradient hooks on all parameters
        
        Args:
            module: PyTorch model to wrap
        """
        super().__init__()
        self.module = module
        
        # Get distributed training info
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # List to store communication handles for async all-reduce operations
        self._handles = []
        
        # Broadcast initial parameters from rank 0 to all other ranks
        # This ensures all ranks start with the same model weights
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        # Register post-accumulate gradient hooks on all parameters
        # These hooks will be called automatically when each parameter's gradient is ready
        for param in self.module.parameters():
            if param.requires_grad:
                def grad_hook(param):
                    handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
                    self._handles.append((handle, param))
                
                param.register_post_accumulate_grad_hook(grad_hook)
    
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
        1. Waits for all pending all-reduce operations to complete
        2. Averages the gradients by dividing by world_size
        3. Clears the handle list for the next iteration
        """
        for handle, param in self._handles:
            # Wait for this all-reduce to complete
            handle.wait()
            # Average the gradient (all_reduce gave us the sum)
            # IMPORTANT: divide AFTER all_reduce, not before
            # Now we update param.grad directly (not the hook's grad argument)
            param.grad.div_(self.world_size)
        
        # Clear handles for next iteration
        self._handles.clear()

