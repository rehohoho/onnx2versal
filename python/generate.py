import argparse

import onnx
from onnx import helper
import onnxruntime

from generate_cpp import CppGenerator
from op_parsers import save_tensor
from check import load_txt


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
  parser.add_argument("onnx",             nargs=1, help="required path to onnx file")
  parser.add_argument("single_input_npy", nargs=1, help="path to single input tensor .npy")
  parser.add_argument("many_input_npy",   nargs=1, help="path to multiple input tensor .npy")
  parser.add_argument("-data", default="../data", help="path to data directory")
  parser.add_argument("-ndata", type=int, default=100, help="number of end to end data points")
  parser.add_argument("-is_output_all", action="store_true", help="whether to output for all layers")
  args = parser.parse_args()
  args.onnx = args.onnx[0]
  args.single_input_npy = args.single_input_npy[0]
  args.many_input_npy = args.many_input_npy[0]
  
  # Update ONNX to output intermediate tensors
  onnx_inter_path = args.onnx.replace(".onnx", "_inter.onnx")
  output_names = generate_inter_graph(onnx_path=args.onnx, 
                                      onnx_inter_path=onnx_inter_path)
  ort_session = onnxruntime.InferenceSession(onnx_inter_path)

  # Load data, shape according to model def, run with ONNX Runtime
  input_shape = ort_session.get_inputs()[0].shape
  input_shape = [i if isinstance(i, int) else -1 for i in input_shape]
  input_dtype = ort_session.get_inputs()[0].type.replace("tensor(", "").strip(")")
  if input_dtype == "float":
    input_dtype = "float32"
  single_input = load_txt(args.single_input_npy).reshape(input_shape).astype(input_dtype)
  many_inputs = load_txt(args.many_input_npy).reshape(input_shape).astype(input_dtype)
  
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
  
  model_input_path = f"{args.data}/{list(cppGenerator.modelin_2_tensor.keys())[0]}_host.txt"
  model_output_path = f"{args.data}/{list(cppGenerator.modelout_2_op.values())[0].name}_goldenout_host.txt"
  save_tensor(model_input_path, many_inputs)
  save_tensor(model_output_path, ort_outs[-1])
