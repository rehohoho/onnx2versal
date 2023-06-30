from typing import List, Mapping, Any
import os

import numpy as np
import onnx
from onnx import numpy_helper
# from google.protobuf.json_format import MessageToJson, MessageToDict

from op_parsers import dtype_to_cstr, save_tensor, OpParser, \
  AddOp, ArgmaxOp, ConvOp, DequantizeLinearOp, GemmOp, MacOp, PoolOp, QGemmOp, \
  QLinearAddOp, QLinearConvOp, QLinearMacOp, QLinearPoolOp, QLinearSoftmaxOp, QuantizeLinearOp, SoftmaxOp, \
  TransposeOp


class Parser:

  def __init__(self,
               data_path: str,
               onnx_path: str,
               data_count: int,
               input_tensors: List[np.ndarray],
               output_tensors: Mapping[str, np.ndarray],
               is_output_all: bool = False):
    self.is_output_all = is_output_all
    self.data_path = data_path
    self.data_count = data_count

    self.op_list: List[OpParser] = []
    self.nodeout_2_adfport: Mapping[str, str] = {}
    self.adf_connects: List[str] = []
    
    model = onnx.load(onnx_path)
    self.nodes = model.graph.node
    # for node in self.nodes:
    #   node_str = MessageToJson(node)
    #   with open("nodes.json", "a") as f:
    #     f.write(node_str)
    self.graph_name = os.path.splitext(os.path.basename(onnx_path))[0]
    
    # register model input, node output and model outputs
    self.modelin_2_tensor = {i.name: t for i, t in zip(model.graph.input, input_tensors)}
    for i, model_input in enumerate(model.graph.input):
      self.nodeout_2_adfport[model_input.name] = f"plin[{i}].out[0]"
    self.modelout_2_op = {i.name: None for i in model.graph.output}

    # save inputs
    for inp_name, input_tensor in self.modelin_2_tensor.items():
      save_tensor(f"{self.data_path}/{self.get_filename(inp_name)}", input_tensor)

    # store I/O tensors and model parameters
    self.input_tensors: List[np.ndarray] = input_tensors
    self.initializers: Mapping[str, np.ndarray] = {
      init.name: init for init in model.graph.initializer}
    self.output_tensors: Mapping[str, np.ndarray] = output_tensors
  
  def get_optype(self, i: int):
    if i >= len(self.nodes):
      return ""
    return self.nodes[i].op_type

  def get_filename(self, filename: str, is_dout: bool = True):
    filename = os.path.splitext(filename)[0]
    suffix = "" if is_dout else "_host"

    fn_list = filename.split("shape")
    if len(fn_list) == 2 and not is_dout:
      shape_list = fn_list[1].split("x")
      shape_list[0] = f"{self.data_count}"
      filename = fn_list[0] + "shape" + "x".join(shape_list)
    
    return f"{filename}{suffix}.txt"

  def get_tensor(self, name: str):
    if name in self.modelin_2_tensor:
      return self.modelin_2_tensor[name]
    elif name in self.initializers:
      return numpy_helper.to_array(self.initializers[name])
    elif name in self.output_tensors:
      return self.output_tensors[name]
    else:
      raise ValueError(f"Unable to find {name} in initializers or output_tensors.")
    
  def register_port(self, 
                    onnx_innames: List[str], 
                    onnx_outnames: List[str], 
                    op: OpParser):
    for i, input_name in enumerate(onnx_innames):
      self.adf_connects.append(op.get_connect_line(self.nodeout_2_adfport[input_name], i))
    gmio_connects = op.get_gmio_connect_line(len(onnx_innames))
    if gmio_connects != "":
      self.adf_connects.append(gmio_connects)
    for i, output_name in enumerate(onnx_outnames):
      self.nodeout_2_adfport[output_name] = f"{op.name}.pout[{i}]"
      if output_name in self.modelout_2_op:
        self.modelout_2_op[output_name] = op

  # assumes nodes are in topological sorted order
  def parse(self):
    i = 0
    while i < len(self.nodes):
      node = self.nodes[i]

      if node.op_type == "Add":
        is_relu = self.get_optype(i+1) == "Relu"
        op = AddOp(f"k{i}add", is_relu)
        
        if is_relu: # lookahead
          print(f"WARNING: fusing Add+Relu")
          i += 1
        
        onnx_out_name = self.nodes[i].output[0]
        op.register_params([self.get_tensor(tname) for tname in (*node.input, onnx_out_name)])
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port(node.input, [onnx_out_name], op)
      
      elif node.op_type == "AveragePool":
        op = PoolOp(f"k{i}pool", reduction_mode="avg")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port(node.input, [onnx_out_name], op)
      
      elif node.op_type == "Conv":
        is_relu = self.get_optype(i+1) == "Relu"
        op = ConvOp(f"k{i}conv", is_relu)
        
        if is_relu: # lookahead
          print(f"WARNING: fusing Conv+Relu")
          i += 1
        
        onnx_out_name = self.nodes[i].output[0]
        op.register_params([self.get_tensor(tname) for tname in (*node.input, onnx_out_name)], node.attribute)
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port([node.input[0]], [onnx_out_name], op)
      
      elif node.op_type == "MaxPool":
        op = PoolOp(f"k{i}pool", reduction_mode="max")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port([node.input[0]], [node.output[0]], op)
      
      elif node.op_type == "Mul":
        if self.get_optype(i+1) != "Add": # lookahead
          raise NotImplementedError("No Add found after Mul, no valid implementation.")
        bias_name = self.nodes[i+1].input[1]
        
        is_relu = self.get_optype(i+2) == "Relu"
        op = MacOp(f"k{i}mac", is_relu=is_relu)

        i += 1 # add
        if is_relu:
          print(f"WARNING: fusing Mul+Add+Relu")
          i += 1 # relu
        else:
          print(f"WARNING: fusing Mul+Add")
        
        onnx_out_name = self.nodes[i].output[0]
        op.register_params([self.get_tensor(tname) for tname in (*node.input, bias_name, onnx_out_name)])
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port([node.input[0]], [onnx_out_name], op)

      elif node.op_type == "Gemm":
        is_relu = self.get_optype(i+1) == "Relu" # lookahead
        op = GemmOp(f"k{i}gemm", is_relu=is_relu)
        
        if self.get_optype(i+1) == "Relu":
          print(f"WARNING: fusing Gemm+Relu")
          i += 1
        
        onnx_out_name = self.nodes[i].output[0]
        op.register_params([self.get_tensor(tname) for tname in (*node.input, onnx_out_name)], 
                           node.attribute)
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port([node.input[0]], [onnx_out_name], op)
        
      elif node.op_type == "QuantizeLinear":
        op = QuantizeLinearOp(f"k{i}quantizelinear")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)])
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port([node.input[0]], [node.output[0]], op)

      elif node.op_type == "QLinearAdd":
        is_relu = self.get_optype(i+1) == "Relu"
        op = QLinearAddOp(f"k{i}qlinearadd", is_relu)
        
        if is_relu: # lookahead
          print(f"WARNING: fusing Add+Relu")
          i += 1
        
        onnx_out_name = self.nodes[i].output[0]
        op.register_params([self.get_tensor(tname) for tname in (*node.input, onnx_out_name)])
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port([node.input[0], node.input[3]], [onnx_out_name], op)

      elif node.op_type == "QLinearConv":
        op = QLinearConvOp(f"k{i}qlinearconv")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port([node.input[0]], [node.output[0]], op)
      
      elif node.op_type == "QLinearAveragePool":
        op = QLinearPoolOp(f"k{i}qlinearpool", reduction_mode="avg")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port([node.input[0]], [node.output[0]], op)
        
      elif node.op_type == "QGemm":
        op = QGemmOp(f"k{i}qgemm")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)],
                           node.attribute)
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port([node.input[0]], [node.output[0]], op)
      
      elif node.op_type == "QLinearMul":
        if self.get_optype(i+1) != "QLinearAdd": # lookahead
          raise NotImplementedError("No QLinearAdd found after QLinearMul, no valid implementation.")
        add_node_inputs = self.nodes[i+1].input
        
        is_relu = self.get_optype(i+2) == "Relu"
        op = QLinearMacOp(f"k{i}qlinearmac", is_relu=is_relu)

        i += 1 # add
        if is_relu:
          print(f"WARNING: fusing QLinearMul+QLinearAdd+Relu")
          i += 1 # relu
        else:
          print(f"WARNING: fusing QLinearMul+QLinearAdd")
        
        onnx_out_name = self.nodes[i].output[0]
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *add_node_inputs, onnx_out_name)])
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port([node.input[0]], [onnx_out_name], op)
      
      elif node.op_type == "DequantizeLinear":
        op = DequantizeLinearOp(f"k{i}dequantizeLinear")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)])
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port([node.input[0]], [node.output[0]], op)

      elif node.op_type in ["Shape", "Constant", "Gather", "Unsqueeze", "Concat", "Reshape"]:
        if len(self.op_list) == 0:
          print(f"Adding output output {node.output[0]} output")
          self.nodeout_2_adfport[node.output[0]] = "plin[0].out[0]"
        elif len(node.output[0]) != 0 and np.all(self.get_tensor(node.output[0]).flatten() == self.op_list[-1].tout.flatten()):
          print(f"Found matching output {node.output[0]} and {op.name} output")
          self.nodeout_2_adfport[node.output[0]] = f"{op.name}.pout[0]"
          self.op_list[-1].disable_output_pad()
        else:
          print(f"WARNING: {node.op_type} not implemented, skipping...")
      
      elif node.op_type == "MatMul":
        if self.get_optype(i+1) != "Add": # lookahead
          raise NotImplementedError("No Add found after MatMul, no valid implementation.")
        bias_name = self.nodes[i+1].input[1]
        
        is_relu = self.get_optype(i+2) == "Relu"
        op = GemmOp(f"k{i}gemm", is_relu=is_relu)
        i += 1
        if is_relu:
          print(f"WARNING: fusing MatMul+Add+Relu")
          i += 1
        else:
          print(f"WARNING: fusing MatMul+Add")
        
        onnx_out_name = self.nodes[i].output[0]
        op.register_params([self.get_tensor(tname) for tname in (*node.input, bias_name, onnx_out_name)],
                           node.attribute)
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port([node.input[0]], [onnx_out_name], op)
      
      elif node.op_type == "Softmax":
        op = SoftmaxOp(f"k{i}softmax")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)])
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port([node.input[0]], [node.output[0]], op)
      
      elif node.op_type == "QLinearSoftmax":
        op = QLinearSoftmaxOp(f"k{i}qlinearsoftmax")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port([node.input[0]], [node.output[0]], op)
      
      elif node.op_type == "Transpose":
        op = TransposeOp(f"k{i}transpose")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port([node.input[0]], [node.output[0]], op)
      
      else:
        raise ValueError(f"Unexpected op_type {node.op_type}")

      if self.is_output_all:
        for ioname in [*node.input, *node.output]:
          tensor = self.get_tensor(ioname)
          tensor_shapestr = "x".join(str(dim) for dim in tensor.shape)
          out_path = f"{i}__{node.name}__{ioname}__{tensor_shapestr}.txt".replace("/", "_")
          out_path = f"{self.data_path}/{out_path}"
          if "weight" in ioname or "bias" in ioname:
            out_name = ioname.replace("/", "_").replace(".", "_")
            tensor_str = str(tensor.flatten().tolist())[1:-2]
            tmp = f"std::vector<{dtype_to_cstr(tensor.dtype)}> {out_name} {{{tensor_str}}};"
            with open(out_path, "w") as f: 
              f.write(tmp)
          else:
            save_tensor(out_path, tensor)
        
      i += 1
      
    self.op_list[0].disable_input_pad()
    self.op_list[-1].disable_output_pad()
