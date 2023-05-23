import numpy as np
import torch
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
import onnx
from onnx import helper
import onnxruntime

from generate_cpp import CppGenerator


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


def get_n_data(dataset: torch.utils.data.Dataset,
               data_count: int):
  data = None
  print(f"Generating MNIST txt for {data_count} data points")
  for x, y in dataset:
    if data is None:
      data = x.unsqueeze(0)
    else:
      data = torch.concat((data, x.unsqueeze(0)))
    if data.shape[0] >= data_count:
      break
  return data


if __name__ == '__main__':
  DATA_COUNT = 100
  DATA_PATH = "../data"
  ONNX_PATH = "../models/lenet_mnist_int8.onnx"
  IS_OUTPUT_ALL_NODES = False

  # Data
  dataset = mnist.MNIST(root=DATA_PATH, train=False, transform=ToTensor(), download=True)
  input_tensor = dataset[0][0].unsqueeze(0)
  
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
                              data_count=DATA_COUNT,
                              input_tensors=[input_tensor.numpy()], 
                              output_tensors=output_tensors, 
                              is_output_all=IS_OUTPUT_ALL_NODES)
  cppGenerator.parse()
  cppGenerator.generate_cpp_graph()
  cppGenerator.generate_xtg_python()
  cppGenerator.generate_cfg()
  cppGenerator.generate_host_cpp()

  # Generate end-to-end data
  data = get_n_data(dataset, DATA_COUNT)
  ort_session = onnxruntime.InferenceSession(ONNX_PATH)
  ort_inputs = {ort_session.get_inputs()[0].name: data.numpy()}
  ort_outs = ort_session.run(None, ort_inputs)
  
  model_input_path = f"{DATA_PATH}/{list(cppGenerator.modelin_2_tensor.keys())[0]}_host.txt"
  model_output_path = f"{DATA_PATH}/{list(cppGenerator.modelout_2_op.values())[0].name}_goldenout_host.txt"
  np.savetxt(model_input_path, data.numpy().reshape(-1, 2), fmt="%.7g")
  np.savetxt(model_output_path, ort_outs[-1].reshape(-1, 2), fmt="%.7g")
