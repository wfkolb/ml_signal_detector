import os
import pickle
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        if("Training" in img_dir):
            with open("TrainingData/training_data.pkl", "rb") as f:
                data = pickle.load(f)
            print('loaded training data')
        if("TestData" in img_dir):
            with open("TestData/Test_data.pkl", "rb") as f:
                data = pickle.load(f)
            print('Loaded test Data')
        self.img_labels = [label for _, label in data]
        #self.img_dir = './TrainingData'
        self.transform = transform
        self.target_transform = target_transform
        self.realData = data

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
       vector, label = self.realData[idx]
       vector = torch.tensor(vector, dtype=torch.float32)
       assert vector.shape[0] ==784 , "Vector does not match intentded size"
       if isinstance(label, str):
            # Example: simple mapping
            label = 1 if label == "Yes" else 0
       label = torch.tensor(label, dtype=torch.float32)
       assert label.shape[0] ==2 , "Vector does not match intentded size"
       #assert vector.shape[0] != 1  
       return  vector, label