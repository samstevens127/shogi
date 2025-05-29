import torch
import os

def save_checkpoint(model, optimizer, iteration, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"✅ Saved checkpoint at {path}")

def load_checkpoint(model, optimizer=None, path=None, map_location='cpu'):
    if path is None or not os.path.exists(path):
        print(f"No checkpoint found at {path}, starting fresh.")
        return 0  # Start at iteration 0

    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint from {path}")
    return checkpoint.get('iteration', 0)
