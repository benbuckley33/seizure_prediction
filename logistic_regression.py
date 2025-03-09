import torch
import torch.nn as nn
import numpy as np

#load data
npzfile = np.load("/Users/benbuckley/Downloads/toy_data_two_moon.npz")
data_train = torch.FloatTensor(npzfile['arr_0'])
label_train = torch.FloatTensor(npzfile['arr_2'])
data_test = torch.FloatTensor(npzfile['arr_1'])
label_test = torch.FloatTensor(npzfile['arr_3'])

#define model and loss/optimizer
model = nn.Linear(2,1)
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

#training loop
epochs = 700
logistic_acc = []

for epoch in range(epochs):
    train = model(data_train)
    loss = loss_function(train, label_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        label_test_hat = model(data_test)
        predicted = (label_test_hat > 0).float()
        accuracy = float(sum(predicted == label_test))/float(label_test.shape[0])
        logistic_acc.append(accuracy)
    if epoch % 20 == 0:
        print(f'Epoch {epoch/epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy*100:.2f}%')
print(f'Prediction accuracy: {logistic_acc[-1]*100}')

