import torch
import torch as nn
import torch as optim

#synthetic data creation
torch.manual_seed(42)
square_footage = torch.linspace(1000,5000,100).view(-1,1)
true_slope = 200
true_intercept = 5000
noise = torch.randn(square_footage.size(0), 1) * 1000

house_prices = square_footage * true_slope + true_intercept + noise
house_prices = house_prices.view(-1,1)

#class creation
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1,1)
    def forward(self, x):
        return self.linear(x)

#define loss/optimizer
loss_function = torch.nn.MSELoss()
model = LinearRegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr =0.0001)

#training loop
epochs = 500
for epoch in range(epochs):
    predicted_price = model(square_footage)
    loss = loss_function(predicted_price, house_prices)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print (f'Epoch: {epoch+1}/{epochs}, Loss:{loss.item()}')
print (f'Predicted Slope: {model.linear.weight.item()}')
print (f'Predicted Intercept: {model.linear.bias.item()}')
