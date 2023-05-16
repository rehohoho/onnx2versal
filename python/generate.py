import numpy as np
import torch
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import onnx
from onnx import helper
import onnxruntime

from generate_cpp import CppGenerator


def generate_txt(loader: DataLoader,
                 data_count: int,
                 output_data_path: str,
                 output_label_path: str):
  data = None
  labels = None

  print(f"Generating MNIST txt for {data_count} data points")
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
  
  np.savetxt(output_data_path, data.numpy().reshape(-1, 2), fmt="%.7g")
  np.savetxt(output_label_path, labels.numpy().reshape(-1, 2), fmt="%.7g")


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
  DATA_COUNT = 100
  DATA_PATH = "../data"
  ONNX_PATH = "../models/lenet_mnist.onnx"
  TEST_DATA_TXT = "../data/mnist_test_data.txt"
  TEST_LABEL_TXT = "../data/mnist_test_label.txt"
  IS_OUTPUT_ALL_NODES = False

  # Data
  dataset = mnist.MNIST(root=DATA_PATH, train=False, transform=ToTensor(), download=True)
  loader = DataLoader(dataset, batch_size=1)
  data, label = next(iter(loader))
  input_tensor = torch.Tensor(data[0]).reshape(1, 1, 28, 28)
  
  # Update ONNX to output intermediate tensors
  onnx_inter_path = ONNX_PATH.replace(".onnx", "_inter.onnx")
  output_names = generate_inter_graph(onnx_path=ONNX_PATH, 
                                      onnx_inter_path=onnx_inter_path)

  # ONNX Runtime
  ort_session = onnxruntime.InferenceSession(onnx_inter_path)
  ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
  ort_outs = ort_session.run(None, ort_inputs)

  # Generate graph info
  output_tensors = {outname: out for outname, out in zip(output_names, ort_outs)}

  cppGenerator = CppGenerator(data_path=DATA_PATH, 
                              onnx_path=ONNX_PATH, 
                              input_tensors=[input_tensor.numpy()], 
                              output_tensors=output_tensors, 
                              is_output_all=IS_OUTPUT_ALL_NODES)
  cppGenerator.parse()
  cppGenerator.generate_cpp_graph()
  cppGenerator.generate_xtg_python()

  # Generate end-to-end data
  generate_txt(loader=loader, 
               data_count=DATA_COUNT, 
               output_data_path=TEST_DATA_TXT, 
               output_label_path=TEST_LABEL_TXT)
