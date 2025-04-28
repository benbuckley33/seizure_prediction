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
        self.x = torch.tensor(x, dtype = torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype = torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y

#cut up data
dataset = SeizureDataset(data, label)
n_total = len(dataset)
n_train = int(0.7 * n_total)
n_test = int(0.2 * n_total)
n_validate = n_total- n_train - n_test
generator = torch.Generator().manual_seed(42)
train, test, validation = torch.utils.data.random_split(dataset = dataset, lengths = [n_train, n_test, n_validate], generator = generator)
batch_size = 128

#dataloader
dataloader_train = DataLoader(dataset = train, batch_size = batch_size, shuffle = True)
dataloader_test = DataLoader(dataset = test, batch_size = batch_size, shuffle = False)

#LSTM class
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size = 5, hidden_size = 128, num_layers = 1, batch_first = True)
        self.dropout = nn.Dropout(p = 0.5)
        self.fc = nn.Linear(128, 2)
    def forward(self, x):
        output, (hn, cn) = self.lstm(x)  
        x = hn[-1]
        x = self.dropout(x)
        x = self.fc(x)
        return x

#loss/optimizer/hyperparameters
num_epoch = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')
model = LSTM().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

#training/testing loop
for epochs in trange(num_epoch, desc = "Epoch", leave = False):
    model.train()
    for (x,y) in tqdm(dataloader_train, desc = "Batch", leave = False):
        x, y = x.to(device), y.to(device)
        predict_y = model(x)
        loss = loss_fn(predict_y, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        test_loss = 0
        correct_pred = 0
        total_pred = 0
        for i, (x,y) in enumerate(tqdm(dataloader_test, desc = "Training", leave = False)):
            x,y = x.to(device), y.to(device)
            y_predict = model(x)
            loss = loss_fn(y_predict, y)
            test_loss += loss
            _, predicted = torch.max(y_predict, 1)
            correct_pred += (predicted == y).sum().item()
            total_pred += y.shape[0]
        accuracy = (correct_pred/total_pred) * 100
        test_loss /= (i+1)

#accuracy and loss
print(f'Epoch: {epochs}, Average Test Loss: {test_loss}')
print(f'Final Accuracy {accuracy}')



            
