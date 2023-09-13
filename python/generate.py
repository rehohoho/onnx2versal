"""
Assumes one input node, easily extendable to many
"""

import argparse
import os

import numpy as np
import onnx
from onnx import helper
import onnxruntime

from parser import Parser
from generator_cpp import CppGenerator
from generator_xtg import XtgGenerator
from generator_cfg import CfgGenerator
from generator_host import HostGenerator
from op_parsers import save_tensor, pad_lastdim, get_shape_str


def get_e2efile_path(path: str, tensor: np.ndarray):
  path, ext = os.path.splitext(path)
  path = path.split("_")
  for i, chunk in enumerate(path):
    if "shape" in chunk:
      path[i] = get_shape_str(tensor)
  return "_".join(path) + ext


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
  parser = argparse.ArgumentParser(description='Generate Versal project files from ONNX model.')
  parser.add_argument("onnx",      nargs=1, help="required path to onnx file")
  parser.add_argument("input_npy", nargs=1, help="path to input data .npy, assume first dim is batch")
  parser.add_argument("-data", default="../data", help="path to data directory")
  parser.add_argument("-ndata", type=int, default=1000, help="number of samples for end to end test")
  parser.add_argument("-is_output_all", action="store_true", help="whether to output for all layers")
  args = parser.parse_args()
  args.onnx = args.onnx[0]
  args.input_npy = args.input_npy[0]
  
  # Update ONNX to output intermediate tensors
  onnx_inter_path = args.onnx.replace(".onnx", "_inter.onnx")
  output_names = generate_inter_graph(onnx_path=args.onnx, 
                                      onnx_inter_path=onnx_inter_path)
  ort_session = onnxruntime.InferenceSession(onnx_inter_path)

  # Load data, shape according to model def, run with ONNX Runtime
  many_inputs = np.load(args.input_npy)[:args.ndata]
  single_input = many_inputs[[0]]
  input_shape = ort_session.get_inputs()[0].shape
  
  if single_input.ndim > len(input_shape):
    single_input = single_input.reshape(-1, *(single_input.shape[-len(input_shape)+1:]))
    many_inputs = many_inputs.reshape(-1, *(many_inputs.shape[-len(input_shape)+1:]))
    single_input = single_input[[0]]
    many_inputs = many_inputs[:args.ndata]
  ort_inputs = {ort_session.get_inputs()[0].name: single_input}
  ort_outs = ort_session.run(None, ort_inputs)

  # Generate graph info
  output_tensors = {outname: out for outname, out in zip(output_names, ort_outs)}

  parser = Parser(data_path=args.data, 
                              onnx_path=args.onnx, 
                              data_count=args.ndata,
                              input_tensors=[single_input], 
                              output_tensors=output_tensors, 
                              is_output_all=args.is_output_all)
  parser.parse()
  CppGenerator(parser).generate_cpp_graph()
  XtgGenerator(parser).generate_xtg_python()
  CfgGenerator(parser).generate_cfg()
  HostGenerator(parser).generate_host_cpp()

  # Generate end-to-end data
  ort_session = onnxruntime.InferenceSession(args.onnx)
  ort_inputs = {ort_session.get_inputs()[0].name: many_inputs}
  ort_outs = ort_session.run(None, ort_inputs)
  
  # save inputs
  for op in parser.g.in_ops:
    input_shape = op.tensor.shape
    input_path = f"{args.data}/{op.get_input_filenames()[0]}"
    input_path = get_e2efile_path(input_path, many_inputs)
    many_inputs = many_inputs.reshape(args.ndata, *input_shape[:-1], -1)
    many_inputs = pad_lastdim(many_inputs, "many inputs", input_shape[-1])
    save_tensor(input_path, many_inputs)

  # save outputs
  for op in parser.g.out_ops:
    for output_filename in op.get_output_filenames(): # can have multiple output per op
      output_path = f"{args.data}/{parser.parse_filename(output_filename, is_e2e=True)}"
      output_path = get_e2efile_path(output_path, ort_outs[-1])
      save_tensor(output_path, ort_outs[-1])

  # Cleanup
  os.remove(onnx_inter_path)
