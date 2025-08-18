import os
import pickle
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class SignalDataSet(Dataset):
    def __init__(self, input_file, img_dir="", transform=None, target_transform=None):
        with open(input_file, "rb") as f:
            data = pickle.load(f)
        print('Loaded Data')
        self.img_labels = [label for _, label in data]
        self.transform = transform
        self.target_transform = target_transform
        self.realData = data

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
       vector, label = self.realData[idx]
       vector = torch.tensor(vector, dtype=torch.float32)
       assert vector.shape[0] ==784 , "Vector does not match intended size"
       label = torch.tensor(label, dtype=torch.float32)
       assert label.shape[0] ==2 , "Vector does not match intended size"
       return  vector, label