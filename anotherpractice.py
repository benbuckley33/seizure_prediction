import numpy as np
import torch
import torch.nn as nn

# Synthetic data (200 samples, 3 features each for training)
x_train = np.random.randn(200, 3) * 2  # Random data with 3 features, spread out
y_train = (x_train[:, 0] + x_train[:, 1] + x_train[:, 2] > 0).astype(int)  # Labels: 1 if sum of features > 0, else 0

# Test data (40 samples, 3 features each)
x_test = np.random.randn(40, 3) * 2
y_test = (x_test[:, 0] + x_test[:, 1] + x_test[:, 2] > 0).astype(int)

# Convert data to torch tensors
data_train = torch.FloatTensor(x_train)
label_train = torch.FloatTensor(y_train).view(-1, 1)
data_test = torch.FloatTensor(x_test)
label_test = torch.FloatTensor(y_test).view(-1, 1)

#define model and loss/optimizer
model =nn.Linear(3,1)
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

#training loop
epochs = 200
accuracy_list = []

for epoch in range(epochs):
    data_train_hat = model(data_train)
    loss = loss_function(data_train_hat, label_train)
    optimizer.zero_grad
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        label_test_hat = model(data_test)
        predictions = torch.sigmoid(label_test_hat) > 0.5
        accurate = predictions == label_test
        accuracy = accurate.sum().float()/label_test.size(0)
        accuracy_list.append(accuracy)
    if epoch % 20 == 0:
        print(f'Epoch: {epoch}, Accuracy: {accuracy}')
print(f' Prediction Accuracy: {accuracy_list[-1].item()}') 
