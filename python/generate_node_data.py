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


def to_numpy(tensor):
  return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def generate_inter_graph(onnx_path: str,
                         onnx_inter_path: str):
  model = onnx.load(onnx_path)
  output_names = [i for node in model.graph.node for i in node.output]
  del model.graph.output[:]
  for outname in output_names:
    intermediate_layer_value_info = helper.ValueInfoProto()
    intermediate_layer_value_info.name = outname
    model.graph.output.append(intermediate_layer_value_info)
  onnx.save(model, onnx_inter_path)
  return output_names


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

  data = np.loadtxt(TEST_DATA_TXT)
  label = np.loadtxt(TEST_LABEL_TXT)
  model_pkl = torch.load(PKL_PATH)
  
  # PyTorch
  input_tensor = torch.Tensor(data[0]).reshape(1, 1, 28, 28)
  torch_out = model_pkl.forward(input_tensor)
  
  # Update ONNX to output intermediate tensors
  output_names = generate_inter_graph(onnx_path=ONNX_PATH, 
                                      onnx_inter_path=ONNX_INTER_PATH)

  # ONNX Runtime  
  ort_session = onnxruntime.InferenceSession(ONNX_INTER_PATH)
  ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
  ort_outs = ort_session.run(None, ort_inputs)

  # compare ONNX Runtime and PyTorch results
  np.testing.assert_allclose(to_numpy(torch_out), ort_outs[-1], rtol=1e-03, atol=1e-05)

  print("Exported model has been tested with ONNXRuntime, and the result looks good!")
