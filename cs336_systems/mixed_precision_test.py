import torch

print("Experiment 1: float32 accumulator, float32 input")
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(f"Result: {s}")

print("\nExperiment 2: float16 accumulator, float16 input")
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(f"Result: {s}")

print("\nExperiment 3: float32 accumulator, float16 input (implicit cast)")
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(f"Result: {s}")

print("\nExperiment 4: float32 accumulator, float16 input (explicit cast)")
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(f"Result: {s}")
