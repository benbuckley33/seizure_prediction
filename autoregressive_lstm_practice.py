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
        self.norm2 = nn.LayerNorm(input_size // 2)
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
        super().__init__()
        seq_data_len = seq_len * 2
        self.input_mlp = nn.Sequential(nn.Linear(seq_data_len, seq_data_len * 4), nn.ELU(), nn.Linear(seq_data_len * 4, 128))
        self.lstm = nn.LSTM(128, 128)
        blocks = [ResBlockMLP(128,128) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*blocks)
        self.fc_out = nn.Linear(128, output_size)
        self.act = nn.ELU()
    def forward(self, input_seq, hidden_in, mem_in):
        input_seq = input_seq.reshape(input_seq.shape[0], -1)
        input_vec = self.input_mlp(input_seq).unsqueeze(0)
        output, (hidden_out, mem_out) = self.lstm(input_vec, (hidden_in, mem_in))
        x = self.act(self.res_blocks(output)).squeeze(0)
        return self.fc_out(x), hidden_out, mem_out
    
#hyperparamters 
weather_lstm = LSTM(seq_len = days_in, output_size= 2)
loss_function = nn.MSELoss()
optimizer = optim.Adam(weather_lstm.parameters(), lr = 1e-4)
nepochs = 500

#training loop
for epoch in trange(nepochs, desc = "Epoch", leave = False):
    weather_lstm.train()
    for day, month, data_seq in tqdm(train_dataloader, desc = "Batch", leave = False):
       seq_block = data_seq[:, :days_in]
       hidden = torch.zeros(1, data_seq.shape[0],128)
       memory = torch.zeros(1, data_seq.shape[0],128)
       loss = 0
       for i in range(day_range - days_in):
           target_seq_block = data_seq[:, i + days_in]
           data_pred, hidden, memory = weather_lstm(seq_block, hidden, memory)
           loss += loss_function(data_pred, target_seq_block)
           seq_block = torch.cat((seq_block[:,1:,:], data_pred.unsqueeze(1).detach()),1)
       loss /= i+1
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
#test loop
data_tensor = torch.FloatTensor(test_data.dataset.values)
log_predictions = []
weather_lstm.eval()
with torch.no_grad():
    seq_block = data_tensor[:, :days_in].unsqueeze(0)
    hidden = torch.zeros(1, seq_block.shape[0], 128)
    memory = torch.zeros(1, seq_block.shape[0], 128)
    for i in range(data_tensor.shape[0] - days_in):
        data_pred, hidden, memory = weather_lstm(seq_block, hidden, memory)
        seq_block = torch.cat((seq_block[:,1:,:], data_pred.unsqueeze(1)),1)
        log_predictions.append(data_pred)
predictions_cat = torch.cat(log_predictions)
un_norm_predictions = (predictions_cat * test_data.std) + test_data.mean
un_norm_data = (data_tensor * test_data.std) + test_data.mean
un_norm_data = un_norm_data[days_in:]

#eval
test_mse = (un_norm_data - un_norm_predictions).pow(2).mean().item()
print("Test MSE value %.2f" % test_mse)
_ = plt.figure(figsize=(10, 5))
_ = plt.plot(un_norm_data[:, 0])
_ = plt.plot(un_norm_predictions[:, 0])
_ = plt.title("Rainfall (mm)")

_ = plt.legend(["Ground Truth", "Prediction"])
_ = plt.figure(figsize=(10, 5))
_ = plt.plot(un_norm_data[:, 1])
_ = plt.plot(un_norm_predictions[:, 1])
_ = plt.title("Max Daily Temperature (C)")

_ = plt.legend(["Ground Truth", "Prediction"])   