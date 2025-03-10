import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader 
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F 
from tqdm import trange, tqdm 
from datasets.Dataset import WeatherDataset

#load data
dataset_file = "/Users/benbuckley/Desktop/master_code/Robo_Seizure/datasets/weather.csv"
split_date = pd.to_datetime('2023-01-01')
day_range = 30
days_in = 14
assert day_range > days_in, "The total day range must be larger than the input days for MLP"
batch_size = 32


#Dataloader
train_data = WeatherDataset(dataset_file, day_range = day_range, split_date=split_date, train_test = "train" )
test_data = WeatherDataset(dataset_file, day_range=day_range, split_date=split_date, train_test= "test")

train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = True)
test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = False, drop_last = True)

#Model
class ResBlockMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.norm2 = nn.Linear(input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, output_size)
        self.fc3 = nn.Linear(input_size, output_size)
        self.act = nn.ELU()
    def forward(self, x):
        x = self.act(self.norm1(x))
        skip = self.fc3(x)
        x = self.act(self.norm2(self.fc1(x)))
        x = self.fc2(x)
        return x + skip
class LSTM(nn.Module):
    def __init__(self, seq_len, output_size, num_blocks = 1):
        seq_data_len = seq_len * 2
        self.input_mlp = nn.Sequential(nn.Linear(seq_data_len, seq_data_len * 4), nn.ELU(), nn.Linear(seq_data_len * 4, 128))
        self.lstm = nn.LSTM(128, 128)
        blocks = [ResBlockMLP(128,128) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*blocks)
        self.fc_out = nn.Linear(128, output_size)
        self.act = nn.ELU
    def forward(self, input_seq, hidden_in, mem_in):
        input_seq = input_seq.reshape(input_seq[0], -1)
        input_vec = self.input_mlp(input_seq).unsqueeze(0)
        output, (hidden_out, mem_out) = self.lstm(input_vec, (hidden_in, mem_in))
        return self.fc_out(x), hidden_out, mem_out
    
#training loop

