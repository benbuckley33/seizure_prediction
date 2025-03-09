import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

#Dataset class

class SinDataset(Dataset):
    def __init__(self, num_datapoints):
        self.x_data = torch.rand(num_datapoints,1)*18-9
        self.y_data = (torch.sin(self.x_data))/2.5
        self.y_data += torch.randn_like(self.y_data)/20
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.x_data.shape[0]

#Train test defining
train_data = 30000
test_data = 8000
batch_size = 256

dataset_train = SinDataset(train_data)
dataset_test = SinDataset(test_data)

dataloader_train = DataLoader(dataset = dataset_train, batch_size = batch_size, shuffle = True)
dataloader_test = DataLoader(dataset = dataset_test, batch_size = batch_size, shuffle = False)

#define forward pass
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)

        x = self.linear2(x)
        x = torch.tanh(x)

        x = self.linear3(x)
        x = torch.tanh(x)

        x = self.linear4(x)
        x = torch.tanh(x)

        return x 

#define loss and optimizer
loss_function = nn.MSELoss()
model = MLP(input_size = 1, hidden_size = 64, output_size =1)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
epochs = 10


#training loop
for epoch in trange(epochs, desc = "Epochs", leave = False):
    for x,y in tqdm(dataloader_train, desc = "Training", leave = False):
        y_predict = model(x)
        loss = loss_function(y_predict, y)
        optimizer.zero_grad
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        test_loss_accum = 0
        for i, (x,y) in enumerate(tqdm(dataloader_test, desc = "Testing", leave = False)):
            y_predict = model(x)
            loss = loss_function(y_predict, y)
            test_loss_accum += loss
        test_loss_accum /= (i+1)
print(f'Epoch: {epoch}, Average Test Loss {test_loss_accum}') 

