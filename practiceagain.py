
import numpy as np
import torch
import torch.nn as nn


# Synthetic data (100 samples, 2 features each)
# Training data
x_train = np.random.randn(100, 2) * 2  # Random data with 2 features, multiplied to spread it out
y_train = (x_train[:, 0] + x_train[:, 1] > 0).astype(int)  # Labels: 1 if the sum of the features > 0, otherwise 0

# Test data (20 samples, 2 features each)
x_test = np.random.randn(20, 2) * 2
y_test = (x_test[:, 0] + x_test[:, 1] > 0).astype(int)

data_train = torch.FloatTensor(x_train)
label_train = torch.FloatTensor(y_train).view(-1,1)
data_test = torch.FloatTensor(x_test)
label_test = torch.FloatTensor(y_test).view(-1,1)

#define model, loss, and optimizer
model = nn.Linear(2,1)
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

#stuff
epochs = 300
accuracy_list = []

#training loop
for epoch in range(epochs):
    data_train_hat = model(data_train)
    loss = loss_function(data_train_hat, label_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #evaluate
    with torch.no_grad():
        label_test_hat = model(data_test)
        predictions = torch.sigmoid(label_test_hat) > 0.5
        accurate = predictions == label_test
        accuracy = accurate.sum().float()/(label_test.size(0))
        accuracy_list.append(accuracy.item())


    if epoch % 20 == 0:
        print(f'Epoch: {epoch} / {epochs}, Loss: {loss}, Accuracy: {(accuracy_list[-1])*100}')
print(f'Prediction Accuracy: {(accuracy_list[-1])*100}')
        
        

