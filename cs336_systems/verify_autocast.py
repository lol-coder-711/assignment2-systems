import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # We will capture intermediate outputs using hooks or just splitting the forward
        pass

device = 'cuda'
model = ToyModel(5, 5).to(device)
# Initial parameters are FP32
print(f"Original fc1 weight dtype: {model.fc1.weight.dtype}")

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

x = torch.randn(2, 5, device=device)
target = torch.randn(2, 5, device=device)

print("\n--- Entering Autocast Context ---")
with torch.autocast(device_type='cuda', dtype=torch.float16):
    # 1. Model parameters (accessing inside context)
    print(f"Parameter dtype inside autocast: {model.fc1.weight.dtype}")
    
    # Forward breakdown
    # x is FP32 initially
    print(f"Input x dtype: {x.dtype}")
    
    out1 = model.fc1(x)
    print(f"fc1 output dtype (Linear): {out1.dtype}")
    
    out_relu = model.relu(out1)
    # ReLU usually preserves dtype
    
    out_ln = model.ln(out_relu)
    print(f"LayerNorm output dtype: {out_ln.dtype}")
    
    logits = model.fc2(out_ln)
    print(f"Logits dtype: {logits.dtype}")
    
    loss = criterion(logits, target)
    print(f"Loss dtype: {loss.dtype}")

# Backward
optimizer.zero_grad()
loss.backward()
print(f"Gradients dtype (fc1.weight.grad): {model.fc1.weight.grad.dtype}")
