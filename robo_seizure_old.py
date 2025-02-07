#import packages
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import butter, filtfilt

#Loading/Preprocessing data
from load_intan_rhd_format import read_data
file_path = '/Users/benbuckley/Downloads/Absence_UCLA_221116_rec-ed_on_221201_221201_125737.rhd'
data = read_data(file_path)
sampling_rate = data['frequency_parameters']['amplifier_sample_rate']
ds_data = data['amplifier_data'][:, ::30]   #downsampling the data so we have a sampling freq of 1KHz

labels_file_path = '/Users/benbuckley/Library/Application Support/MathWorks/MATLAB Add-Ons/Toolboxes/xolotl/examples/seizure_labels.csv'
seizure_labels_df = pd.read_csv(labels_file_path)
print(seizure_labels_df.head())

#Transform into tensors and then into a Dataset object for DataLoader
train_data = torch.tensor(ds_data, dtype=torch.float32)
train_labels = torch.tensor(seizure_labels_df, dtype=torch.float32).view(-1,1)

class SeizureDataset(torch.utils.data.Dataset):
    def __init__(self, train_data, train_labels):
        self.data = train_data
        self.labels = train_labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
    
#Run DataLoader
train_dataset = SeizureDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)

#RNN Model
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNClassifier, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size, batch_first = True)
    self.fc = nn.Linear(hidden_size, 1)
    self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        rnn_out, hidden = self.rnn(x)
        last_hidden_state = rnn_out[:, -1, :]
        fc_out = self.fc(last_hidden_state)
        output = self.sigmoid(fc_out)
        return output
    
# Initializing Parameters
input_size = 256
hidden_size = 64
num_layers = 2
model = RNNClassifier(input_size, hidden_size, num_layers)

#Loss Function/Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = .001)

#Training Loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss/len(train_loader)
    print(f"Epoch [{epoch +1}/{num_epochs}], Loss {avg_loss:.4f}")
 