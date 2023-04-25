from typing import List, Mapping
import os
import json

import numpy as np
import torch
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import onnx
from onnx import helper
from onnx import numpy_helper
import onnxruntime
from google.protobuf.json_format import MessageToDict


def generate_txt(is_train: bool,
                 data_count: int,
                 data_path: str,
                 output_data_path: str,
                 output_label_path: str):
  dataset = mnist.MNIST(root=data_path, train=is_train, transform=ToTensor(), download=True)
  loader = DataLoader(dataset, batch_size=1)
  data = None
  labels = None

  print(f"Generating MNIST txt for train={is_train}")
  for x, y in loader:
    if data is None:
      data = x
    else:
      data = torch.concat((data, x))
    
    if labels is None:
      labels = y
    else:
      labels = torch.concat((labels, y))
    
    if data.shape[0] >= data_count:
      break
  
  np.savetxt(output_data_path, data.numpy().reshape(data.shape[0], -1), fmt="%.7g")
  np.savetxt(output_label_path, labels.numpy().reshape(data.shape[0], -1), fmt="%.7g")


if __name__ == '__main__':
  TXT_DATA_COUNT = 100
  DATA_PATH = "../data"
  PKL_PATH = "../models/lenet_mnist.pkl"
  ONNX_PATH = "../models/lenet_mnist.onnx"
  ONNX_INTER_PATH = "../models/lenet_mnist_inter.onnx"
  TEST_DATA_TXT = "../data/mnist_test_data.txt"
  TEST_LABEL_TXT = "../data/mnist_test_label.txt"
  INTER_TXT_PREFIX = "../data/lenet_mnist"
  OUTPUT_JSON = False
  OUTPUT_TXT = True

  if not os.path.exists(TEST_DATA_TXT) or not os.path.exists(TEST_LABEL_TXT):
    generate_txt(is_train=False, data_count=TXT_DATA_COUNT, data_path=DATA_PATH, 
                output_data_path=TEST_DATA_TXT, output_label_path=TEST_LABEL_TXT)
