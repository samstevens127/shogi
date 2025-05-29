import torch
from torch.utils.data import Dataset

class ShogiDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples  # List of (x, pi, v)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, pi, v = self.samples[idx]
        return x, torch.tensor(pi, dtype=torch.float32), torch.tensor([v], dtype=torch.float32)
