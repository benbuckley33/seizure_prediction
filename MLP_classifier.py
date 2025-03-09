import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import trange, tqdm
import numpy as np
import time

batch_size = 256
data_set_root = "/Users/benbuckley/Desktop/master_code/Robo_Seizure/datasets"
train = MNIST(data_set_root, train = True, download = True, transform = transforms.ToTensor())
test = MNIST(data_set_root, train = False, download = True, transform = transforms.ToTensor())

train_loader = DataLoader(train, batch_size, shuffle = True)
test_loader = DataLoader(test, batch_size, shuffle = False)


#define class
class MLP_classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256 , 128)
        self.fc4 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x= F.relu(self.fc3(x))
        x= F.relu(self.fc4(x))
        return x

#loss/optimizer/call model
num_classes = 10
model = MLP_classifier(num_classes)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= .01)
epochs = 20 

#training loop
def train_loop(model, train_loader, loss_function, optimizer, loss_logger):
    for i, (x,y) in enumerate(tqdm(train_loader, desc = "Training", leave = False)):
        y_train_hat = model(x)
        train_loss = loss_function(y_train_hat, y.long())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        loss_logger.append(train_loss.item())
    return model, loss_logger, optimizer

#test loop
def test_loop(model, test_loader, loss_function, optimizer):
    with torch.no_grad():
        correct_pred = 0
        pred = 0
        total_pred = 0
        for i, (x,y) in enumerate(tqdm(test_loader, desc = "Testing", leave = False)):
            y_train_hat = model(x)
            _, predicted = torch.max(y_train_hat, 1)
            correct_pred += (predicted == y ).sum().item()
            total_pred += y.shape[0]
            accuracy = (correct_pred/total_pred) * 100
        return accuracy


train_loss = []
test_loss = []
test_acc = []
#iterate over all data
for i in trange(epochs, desc = "Epochs", leave = False):
    model, train_loss, optimizer = train_loop(model, train_loader, loss_function, optimizer, train_loss)
    accuracy = test_loop(model, test_loader, loss_function, optimizer)
    test_acc.append(accuracy)
print (f'Final Accuracy: {accuracy}')
