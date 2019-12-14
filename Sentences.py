from torch.utils.data import Dataset
import torch
import math


class Sentences(Dataset):
    def __init__(self, data, F2I, L2I, window_size):
        self.data = data
        self.F2I = F2I
        self.L2I = L2I
        self.window_size = window_size

    def __len__(self):
        return len(self.F2I)

    def __getitem__(self, item):
        window = []
        for i in range(self.window_size):
            window.append(torch.tensor(self.F2I[item + i][0]))
        return window, torch.tensor(self.L2I[self.data[item + math.floor(float(self.window_size) / 2)][1]], dtype=torch.long)
