import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from tqdm import trange, tqdm


data = np.load("/media/ben2X/SeizureBot/data_vector.npy")
label = np.load("/media/ben2X/SeizureBot/label_vector.npy")

#dataset class
class SeizureDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(data)
        self.y = torch.tensor(label)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y

#cut up data
train = .7(len(label))
test = .2(len(label))
validation = .1(len(label))
batch_size = 256

#dataloader
dataset_train = SeizureDataset(train)
dataset_test = SeizureDataset(test)
dataloader_train = DataLoader(dataset = dataset_train, batch_size = batch_size, shuffle = True)
dataloader_test = DataLoader(dataset = dataset_test, batch_size = batch_size, shuffle = False)

#LSTM class
class LSTM(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(input_size = 5, hidden_size = 128, num_layers = 1)
        self.dropout = nn.Dropout(p = 0.5)
        self.fc = nn.Linear(128, 2)
    def forward(self, x, output, hn, cn):
        output, (hn, cn) = self.lstm(x)   #not sure if this is correct
        x = hn[-1]
        x = self.dropout(x)
        x = self.fc(x)
        return x

#loss/optimizer



#training/testing loop
for 