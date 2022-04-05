import os
import numpy as np
from torch.utils.data import Dataset
import torch
from scipy.signal import butter, lfilter

class EEGLoader(Dataset):
    def __init__(self, data, device, supervised):
        X_data, y_data = data

        self.x_data = torch.Tensor(X_data).to(device)
        self.y_data = torch.Tensor(y_data).to(device)
        self.supervised = supervised

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
        if self.supervised:
            return self.x_data[index,:], self.y_data[index,:]
        else:
            return self.x_data[index,:]

    def getallitem(self):
        if self.supervised:
            return self.x_data, self.y_data
        else:
            return self.x_data

