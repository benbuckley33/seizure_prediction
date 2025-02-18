import torch
import numpy as np

#synthetic data
torch.manual_seed(42)
x = torch.randn(100,3)
true_w = torch.tensor([2.5,-1.2,3.8])
true_b = 0.5

y = x @ true_w +true_b + torch.randn(100) * 0.1

#define parameter
input_size = 3
output_size = 1

#defining class
class LinearRegression (torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(output_size, input_size))
        self.b = torch.nn.Parameter(torch.randn(1, output_size))
    
    def forward(self, x):
        return torch.matmul(x, self.w.t(),) + self.b

#define loss/optimizer
model = LinearRegression(input_size, output_size)
loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

#training loop
epochs = 1000
for epoch in range(epochs):
    y_prediction = model(x)
    current_loss = loss(y_prediction.squeeze(), y)
    optimizer.zero_grad()
    current_loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {current_loss.item():.4f}")
    
print("\nLearned Weights:", model.w.data.squeeze().tolist())
print("Learned Bias:", model.b.data.squeeze().item())