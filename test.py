import torch

print(f"CUDA tilgængelig: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Enhed: {torch.cuda.get_device_name(0)}")
    print(f"Antal enheder: {torch.cuda.device_count()}")