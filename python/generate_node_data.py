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


def get_shape_str(arr: np.ndarray):
  return "x".join(str(dim) for dim in arr.shape)


def get_tensor(name: str,
               input_tensor: np.ndarray,
               initializers: Mapping[str, np.ndarray],
               output_tensors: Mapping[str, np.ndarray]):
  if name == "input":
    return to_numpy(input_tensor)
  elif name in initializers:
    return numpy_helper.to_array(initializers[name])
  elif name in output_tensors:
    return output_tensors[name]
  else:
    raise ValueError(f"Unable to find {name} in initializers or output_tensors.")


def process_gemm_weights(weights: np.ndarray):
  """Expects NxK weights as per PyTorch
  Returns KxN weights, with N padded so N%4=0
  """
  weights = weights.transpose(1,0)
  K, N = weights.shape
  n_pad = (4 - N%4) % 4
  if n_pad != 0:
    print(f"Padding Gemm weights {K, N} to {K, N+n_pad}")
    weights = np.pad(weights, ((0,0),(0,n_pad)), "constant", constant_values=0)
  return weights


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
  OUTPUT_IMAGE = False
  N_PER_ROW = 2 # bitwidth of 64, float32

  dataset = mnist.MNIST(root=DATA_PATH, train=False, transform=ToTensor(), download=True)
  loader = DataLoader(dataset, batch_size=1)

  if not os.path.exists(TEST_DATA_TXT) or not os.path.exists(TEST_LABEL_TXT):
    generate_txt(loader=loader, data_count=TXT_DATA_COUNT, output_data_path=TEST_DATA_TXT, output_label_path=TEST_LABEL_TXT)

  data, label = next(iter(loader))
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

  # generate graph info
  model = onnx.load(ONNX_PATH)
  nodes = model.graph.node
  initializers = {init.name: init for init in model.graph.initializer}
  output_tensors = {outname: out for outname, out in zip(output_names, ort_outs)}

  res = []
  node = nodes[0]
  for i, node in enumerate(nodes):
    node_name = node.name.replace("/", "_").replace(".", "_")
    out_dict = MessageToDict(node)
    for name in out_dict.get("input", []) + out_dict.get("output", []):
      tensor = get_tensor(name, input_tensor, initializers, output_tensors)
      
      key = f"__{name}_{get_shape_str(tensor)}"
      out_dict[key] = " ".join(str(i) for i in tensor.flatten().tolist())
      
      out_name = name.replace("/", "_").replace(".", "_")
      out_txt_path = f"{INTER_TXT_PREFIX}__{i}__{node_name}__{out_name}__{get_shape_str(tensor)}.txt"

      # if tensor.ndim == 4:
      #   tensor = tensor.transpose(0, 2, 3, 1) # BCHW to BHWC
      if "Gemm" in node_name and "weight" in name:
        tensor = process_gemm_weights(tensor)
        import ipdb;ipdb.set_trace()
      
      if "weight" in name or "bias" in name:
        tensor_list = tensor.flatten().tolist()
        if str(tensor.dtype) == "float32":
          tmp = f"std::vector<float> {out_name} {{{str(tensor_list)[1:-2]}}};"
          with open(out_txt_path, "w") as f:
            f.write(tmp)
        else:
          raise ValueError(f"Unsupported type for {name}, {tensor.dtype}.")
      else:
        if tensor.size >= N_PER_ROW:
          np.savetxt(out_txt_path, tensor.reshape(-1, N_PER_ROW))
        else:
          np.savetxt(out_txt_path, tensor.flatten())
      print(f"Exported node: {node_name}; I/O name: {name}; to {out_txt_path}")
    
    if OUTPUT_JSON:
      out_json_path = f"{INTER_TXT_PREFIX}__{i}__{node_name}.json"
      with open(out_json_path, "w") as f:
        json.dump(out_dict, f)  
      print(f"Exported intermediate node {node.name} to {out_json_path}")

  if OUTPUT_IMAGE:
    from torchvision.utils import save_image
    save_image(input_tensor, "input_tensor.png")