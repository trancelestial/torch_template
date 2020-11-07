import torch

class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.points = torch.tensor(...)
    
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, idx):
        return self.points[idx]

