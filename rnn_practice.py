import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from datasets.Dataset import WeatherDataset
import torch.nn.functional as F
from tqdm import tqdm, trange
import pandas as pd
import matplotlib.pyplot as plt


dataset_file = "/Users/benbuckley/Desktop/master_code/Robo_Seizure/datasets/weather.csv"
split_date = pd.to_datetime('2023-01-01')
day_range = 15
days_in = 14
assert day_range > days_in, "the total day range must be larger than the input days for the MLP"

#hyperparameters
learning_rate = 1e-4
nepochs = 500
batch_size = 32 

#dataloader
dataset_train = WeatherDataset(dataset_file, day_range =day_range, split_date = split_date, train_test = "train")
dataset_test = WeatherDataset(dataset_file, day_range =day_range, split_date = split_date, train_test = "test")
data_loader_train = DataLoader(dataset = dataset_train, batch_size = batch_size, shuffle = True, drop_last = True)
data_loader_test = DataLoader(dataset = dataset_test, batch_size = batch_size, shuffle = False, drop_last = True)

#defining the layers of the rnn 
class ResBlockMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_size)
        self.linear1 = nn.Linear(input_size, input_size // 2)
        self.norm2 = nn.LayerNorm(input_size // 2)
        self.linear2 = nn.Linear(input_size // 2, output_size)
        self.linear3 = nn.Linear(input_size, output_size)
        self.act = nn.ELU()
    def forward(self, x):
        x = self.act(self.norm1(x))
        skip = self.linear3(x)
        x = self.act(self.norm2(self.linear1(x)))
        x = self.act(self.linear2(x))
        return x + skip
class RNN(nn.Module):
    def __init__(self, seq_len, output_size, num_blocks = 1, buffer_size = 128):
        super().__init__()
        seq_data_len = seq_len * 2
        self.input_MLP = nn.Sequential(
            nn.Linear(seq_data_len, seq_data_len * 4), nn.ELU(), nn.Linear(seq_data_len * 4, 128), nn.ELU()
        )
        self.rnn = nn.Linear(256, 128)
        blocks = [ResBlockMLP(128,128) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*blocks)
        self.linear_final = nn.Linear(128, output_size)
        self.linear_buffer = nn.Linear(128, buffer_size)
        self.act =nn.ELU()
    def forward(self, input_seq, buffer_in):
        input_seq = input_seq.reshape(input_seq.shape[0], -1)
        input_vec = self.input_MLP(input_seq)
        x_cat = torch.cat((buffer_in, input_vec),1)
        x = self.rnn(x_cat)
        x = self.act(self.res_blocks(x))
        return self.linear_final(x), torch.tanh(self.linear_buffer(x))
    
#loss/optimizer etc.
buffer_size = 128
loss_function = torch.nn.MSELoss()
weather_rnn = RNN(seq_len = days_in, output_size = 2, buffer_size = buffer_size)
optimizer = optim.Adam(weather_rnn.parameters(), lr = learning_rate)
training_loss_logger = []

#train the stuff
for epoch in trange(nepochs, desc = "Epoch", leave = False):
    weather_rnn.train()
    for day, month, data_seq in tqdm(data_loader_train, desc = "Training", leave = False):
        seq_block = data_seq[:, :days_in]
        buffer = torch.zeros(data_seq.shape[0], buffer_size)
        loss = 0
        for i in range (day_range - days_in):
            target_seq_block = data_seq[:, i + days_in]
            data_pred, buffer = weather_rnn(seq_block, buffer)
            loss += loss_function(data_pred, target_seq_block)
            seq_block = torch.cat((seq_block[:,1:,:], data_pred.unsqueeze(1).detach()),1)
        loss /= i + 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss_logger.append(loss.item())

#test the stuff
data_tensor = torch.FloatTensor(dataset_test.dataset.values)
log_predictions = []
weather_rnn.eval()
with torch.no_grad():
    seq_block = data_tensor[:days_in,:].unsqueeze(0)
    buffer = torch.zeros(seq_block.shape[0], buffer_size)
    for i in range(data_tensor.shape[0] - days_in):
        data_pred , buffer = weather_rnn(seq_block, buffer)
        log_predictions.append(data_pred)
        seq_block = torch.cat((seq_block[:,1:,:], data_pred.unsqueeze(1)), 1)
predictions_cat = torch.cat(log_predictions)
un_norm_predictions = (predictions_cat * dataset_test.std) +  dataset_test.mean
un_norm_data = (data_tensor * dataset_test.std) + dataset_test.mean
un_norm_data = un_norm_data[days_in:]                

plt.figure(figsize=(10, 6))
plt.plot(un_norm_data.numpy(), label='True Data')
plt.plot(un_norm_predictions.numpy(), label='Predictions')
plt.legend()
plt.show()