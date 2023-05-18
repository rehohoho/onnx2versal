import torch
from torch import nn


class Lenet(nn.Module):
  def __init__(self):
    super(Lenet, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.relu2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(2)
    self.fc1 = nn.Linear(256, 120)
    self.relu3 = nn.ReLU()
    self.fc2 = nn.Linear(120, 84)
    self.relu4 = nn.ReLU()
    self.fc3 = nn.Linear(84, 10)
    self.relu5 = nn.ReLU()

  def forward(self, x):
    y = self.conv1(x)
    y = self.relu1(y)
    y = self.pool1(y)
    y = self.conv2(y)
    y = self.relu2(y)
    y = self.pool2(y)
    y = y.reshape(y.shape[0], -1)
    y = self.fc1(y)
    y = self.relu3(y)
    y = self.fc2(y)
    y = self.relu4(y)
    y = self.fc3(y)
    y = self.relu5(y)
    return y


class QuantizedLenet(nn.Module):
  def __init__(self):
    super(QuantizedLenet, self).__init__()
    # QuantStub converts tensors from floating point to quantized
    self.quant = torch.ao.quantization.QuantStub()
    # DeQuantStub converts tensors from quantized to floating point
    self.dequant = torch.ao.quantization.DeQuantStub()

    self.conv1 = nn.Conv2d(1, 6, 5)
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.relu2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(2)
    self.fc1 = nn.Linear(256, 120)
    self.relu3 = nn.ReLU()
    self.fc2 = nn.Linear(120, 84)
    self.relu4 = nn.ReLU()
    self.fc3 = nn.Linear(84, 10)
    self.relu5 = nn.ReLU()

  def forward(self, x):
    # manually specify where tensors will be converted from floating
    # point to quantized in the quantized model
    x = self.quant(x)

    x = self.conv1(x)
    x = self.relu1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.pool2(x)
    
    # manually specify where tensors will be converted from quantized
    # to floating point in the quantized model
    x = self.dequant(x)
    
    x = x.reshape(x.shape[0], -1)
    
    x = self.quant(x)
    
    x = self.fc1(x)
    x = self.relu3(x)
    x = self.fc2(x)
    x = self.relu4(x)
    x = self.fc3(x)
    x = self.relu5(x)
    
    x = self.dequant(x)
    
    return x
