import argparse
import sys

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
  parser = argparse.ArgumentParser(description='Generate Versal project files from ONNX model.')
  parser.add_argument("onnx", nargs=1, type=str, help="required path to onnx file")
  parser.add_argument("-data", type=str, default="../data", help="path to data directory")
  parser.add_argument("-ndata", type=int, default=100, help="number of end to end data points")
  parser.add_argument("-is_output_all", action="store_true", help="whether to output for all layers")
  args = parser.parse_args()
  args.onnx = args.onnx[0]

  # Data
  dataset = mnist.MNIST(root=args.data, train=False, transform=ToTensor(), download=True)
  input_tensor = dataset[0][0].unsqueeze(0)
  
  # Update ONNX to output intermediate tensors
  onnx_inter_path = args.onnx.replace(".onnx", "_inter.onnx")
  output_names = generate_inter_graph(onnx_path=args.onnx, 
                                      onnx_inter_path=onnx_inter_path)

  # ONNX Runtime
  ort_session = onnxruntime.InferenceSession(onnx_inter_path)
  ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
  ort_outs = ort_session.run(None, ort_inputs)

  # Generate graph info
  output_tensors = {outname: out for outname, out in zip(output_names, ort_outs)}

  cppGenerator = CppGenerator(data_path=args.data, 
                              onnx_path=args.onnx, 
                              data_count=args.ndata,
                              input_tensors=[input_tensor.numpy()], 
                              output_tensors=output_tensors, 
                              is_output_all=args.is_output_all)
  cppGenerator.parse()
  cppGenerator.generate_cpp_graph()
  cppGenerator.generate_xtg_python()
  cppGenerator.generate_cfg()
  cppGenerator.generate_host_cpp()

  # Generate end-to-end data
  data = get_n_data(dataset, args.ndata)
  ort_session = onnxruntime.InferenceSession(args.onnx)
  ort_inputs = {ort_session.get_inputs()[0].name: data.numpy()}
  ort_outs = ort_session.run(None, ort_inputs)
  
  model_input_path = f"{args.data}/{list(cppGenerator.modelin_2_tensor.keys())[0]}_host.txt"
  model_output_path = f"{args.data}/{list(cppGenerator.modelout_2_op.values())[0].name}_goldenout_host.txt"
  np.savetxt(model_input_path, data.numpy().reshape(-1, 2), fmt="%.7g")
  np.savetxt(model_output_path, ort_outs[-1].reshape(-1, 2), fmt="%.7g")
