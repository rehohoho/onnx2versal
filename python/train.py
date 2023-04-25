import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
import torch.onnx

from model import Lenet


if __name__ == "__main__":
  BATCH_SIZE = 256
  ALL_EPOCH = 1
  DATA_PATH = "../data"
  MODEL_PATH = "../models"

  device = "cuda" if torch.cuda.is_available() else "cpu"
  train_dataset = mnist.MNIST(root=DATA_PATH, train=True, transform=ToTensor(), download=True)
  test_dataset = mnist.MNIST(root=DATA_PATH, train=False, transform=ToTensor(), download=True)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
  model = Lenet().to(device)
  sgd = SGD(model.parameters(), lr=1e-1)
  loss_fn = CrossEntropyLoss()
  prev_acc = 0

  for current_epoch in range(ALL_EPOCH):
    model.train()
    for idx, (train_x, train_label) in enumerate(train_loader):
      train_x = train_x.to(device)
      train_label = train_label.to(device)
      sgd.zero_grad()
      predict_y = model(train_x.float())
      loss = loss_fn(predict_y, train_label.long())
      loss.backward()
      sgd.step()

    all_correct_num = 0
    all_sample_num = 0
    model.eval()
    
    for idx, (test_x, test_label) in enumerate(test_loader):
      test_x = test_x.to(device)
      test_label = test_label.to(device)
      predict_y = model(test_x.float()).detach()
      predict_y =torch.argmax(predict_y, dim=-1)
      current_correct_num = predict_y == test_label
      all_correct_num += np.sum(current_correct_num.to("cpu").numpy(), axis=-1)
      all_sample_num += current_correct_num.shape[0]

    acc = all_correct_num / all_sample_num
    print(f"accuracy: {acc:.3f}", flush=True)
    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    torch.save(model, f"{MODEL_PATH}/lenet_mnist.pkl")
    
    if np.abs(acc - prev_acc) < 1e-4:
        break  
    prev_acc = acc
  
  print("Model finished training")

  torch.onnx.export(
    model,                      # model being run
    test_x,                     # model input (or a tuple for multiple inputs)
    f"{MODEL_PATH}/lenet_mnist.onnx", # where to save the model (can be a file or file-like object)
    export_params=True,         # store the trained parameter weights inside the model file
    # opset_version=10,         # the ONNX version to export the model to
    # do_constant_folding=True, # whether to execute constant folding for optimization
    input_names = ["input"],    # the model's input names
    output_names = ["output"],  # the model's output names
    dynamic_axes={"input" : {0 : "batch_size"},    # variable length axes
                  "output" : {0 : "batch_size"}})
  print("Converted to onnx")