import argparse

import numpy as np
import onnx
from onnx import helper
import onnxruntime

from generate_cpp import CppGenerator
from op_parsers import save_tensor


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
  parser.add_argument("-ndata", default=1000, help="number of samples for end to end test")
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
  
  ort_inputs = {ort_session.get_inputs()[0].name: single_input}
  ort_outs = ort_session.run(None, ort_inputs)

  # Generate graph info
  output_tensors = {outname: out for outname, out in zip(output_names, ort_outs)}

  cppGenerator = CppGenerator(data_path=args.data, 
                              onnx_path=args.onnx, 
                              data_count=args.ndata,
                              input_tensors=[single_input], 
                              output_tensors=output_tensors, 
                              is_output_all=args.is_output_all)
  cppGenerator.parse()
  cppGenerator.generate_cpp_graph()
  cppGenerator.generate_xtg_python()
  cppGenerator.generate_cfg()
  cppGenerator.generate_host_cpp()

  # Generate end-to-end data
  ort_session = onnxruntime.InferenceSession(args.onnx)
  ort_inputs = {ort_session.get_inputs()[0].name: many_inputs}
  ort_outs = ort_session.run(None, ort_inputs)
  
  model_input_path = f"{args.data}/host_{list(cppGenerator.modelin_2_tensor.keys())[0]}.txt"
  model_output_path = f"{args.data}/host_{list(cppGenerator.modelout_2_op.keys())[0]}.txt"
  save_tensor(model_input_path, many_inputs)
  save_tensor(model_output_path, ort_outs[-1])
