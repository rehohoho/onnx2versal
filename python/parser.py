from typing import List, Mapping, Any
from collections import OrderedDict
import os

import numpy as np
import onnx
from onnx import numpy_helper
from google.protobuf.json_format import MessageToJson, MessageToDict

from op_parsers import dtype_to_cstr, save_tensor, OpParser, \
  AddOp, ArgmaxOp, ConvOp, DequantizeLinearOp, GemmOp, IdentityOp, InputOp, MacOp, PoolOp, QGemmOp, \
  QLinearAddOp, QLinearConvOp, QLinearMacOp, QLinearPoolOp, QLinearSoftmaxOp, QuantizeLinearOp, SoftmaxOp, \
  TransposeOp

SKIPPABLE_NODES = ["Shape", "Constant", "Gather", "Unsqueeze", "Concat", "Reshape", "Flatten"]

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

    self.onnxname_2_op: Mapping[str, OpParser] = OrderedDict()
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
      self.onnxname_2_op[model_input.name] = InputOp(model_input.name, i)
    self.modelout_2_op = {i.name: None for i in model.graph.output}

    # store I/O tensors and model parameters
    self.input_tensors: List[np.ndarray] = input_tensors
    self.initializers: Mapping[str, np.ndarray] = {
      init.name: init for init in model.graph.initializer}
    self.output_tensors: Mapping[str, np.ndarray] = output_tensors

    self.gmiobuf_2_size = {}
  
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
  
  def get_input_filename(self, is_dout: bool) -> List[str]:
    input_filenames = [i for i in self.modelin_2_tensor.keys()]
    return [self.get_filename(fn, is_dout) for fn in input_filenames]

  def get_output_filename(self, is_dout: bool) -> List[str]:
    output_filenames = [op.get_output_filename() for op in self.modelout_2_op.values()]
    return [self.get_filename(fn, is_dout) for fn in output_filenames]
  
  def get_input_id_name_tensor(self):
    for i, (name, tensor) in enumerate(self.modelin_2_tensor.items()):
      yield i, name, tensor

  def get_output_id_op(self, include_output: bool = True, include_optional_output: bool = False):
    i = 0
    ops = []
    if include_output:
      for op in self.modelout_2_op.values():
        ops.append((i, op))
        i += 1
    if include_optional_output:
      for op in self.onnxname_2_op.values():
        if op not in self.modelout_2_op.values():
          ops.append((i, op))
          i += 1
    for i, op in ops:
      if isinstance(op, InputOp): continue # skip input node
      yield i, op

  def register_port(self, 
                    onnx_innames: List[str], 
                    onnx_outnames: List[str], 
                    op: OpParser):
    for i, input_name in enumerate(onnx_innames):
      in_port = self.onnxname_2_op[input_name].get_adf_port_name()
      self.adf_connects.append(op.get_connect_line(in_port, i))
    
    gmio_connects = op.get_gmio_connect_line(len(onnx_innames))
    if gmio_connects != "":
      self.adf_connects.append(gmio_connects)
    
    for i, output_name in enumerate(onnx_outnames):
      if output_name in self.modelout_2_op:
        self.modelout_2_op[output_name] = op

  # assumes nodes are in topological sorted order
  def parse(self):
    i = 0
    while i < len(self.nodes):
      node = self.nodes[i]

      if node.op_type in SKIPPABLE_NODES:
        last_op = next(reversed(self.onnxname_2_op.values()))
        node_output_name = node.output[0]
        if len(node_output_name) != 0 and np.all(self.get_tensor(node_output_name).flatten() == last_op.tout.flatten()):
          print(f"Found matching output {node_output_name} and {op.name} output")
          del self.onnxname_2_op[next(reversed(self.onnxname_2_op))]
          self.onnxname_2_op[node_output_name] = last_op
          last_op.disable_output_pad()
        else:
          print(f"WARNING: {node.op_type} not implemented, skipping...")
      
      elif node.op_type == "DequantizeLinear" and self.get_optype(i+1) in SKIPPABLE_NODES and self.get_optype(i+2) == "QuantizeLinear":
        last_op = next(reversed(self.onnxname_2_op.values()))
        node_output_name = self.nodes[i+2].output[0]
        if np.all(self.get_tensor(node_output_name).flatten() == last_op.tout.flatten()):
          print(f"Found matching output {node_output_name} and {op.name} output")
          del self.onnxname_2_op[next(reversed(self.onnxname_2_op))]
          self.onnxname_2_op[node_output_name] = last_op
          last_op.disable_output_pad()
        else:
          raise ValueError(f"Dequantize-{self.get_optype(i+1)}-Quantize yield different result, not skippable.")
        i += 2
      
      elif node.op_type == "Add":
        is_relu = self.get_optype(i+1) == "Relu"
        op = AddOp(f"k{i:03d}add", is_relu)
        
        if is_relu: # lookahead
          print(f"WARNING: fusing Add+Relu")
          i += 1
        
        onnx_out_name = self.nodes[i].output[0]
        op.register_params([self.get_tensor(tname) for tname in (*node.input, onnx_out_name)])
        op.save_txt(self.data_path)
        self.onnxname_2_op[onnx_out_name] = op
        self.register_port(node.input, [onnx_out_name], op)
      
      elif node.op_type == "AveragePool":
        op = PoolOp(f"k{i:03d}pool", reduction_mode="avg")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.onnxname_2_op[onnx_out_name] = op
        self.register_port(node.input, [onnx_out_name], op)
      
      elif node.op_type == "Conv":
        is_relu = self.get_optype(i+1) == "Relu"
        op = ConvOp(f"k{i:03d}conv", is_relu)
        
        if is_relu: # lookahead
          print(f"WARNING: fusing Conv+Relu")
          i += 1
        
        onnx_out_name = self.nodes[i].output[0]
        op.register_params([self.get_tensor(tname) for tname in (*node.input, onnx_out_name)], node.attribute)
        op.save_txt(self.data_path)
        self.onnxname_2_op[onnx_out_name] = op
        self.register_port([node.input[0]], [onnx_out_name], op)
      
      elif node.op_type == "MaxPool":
        op = PoolOp(f"k{i:03d}pool", reduction_mode="max")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.onnxname_2_op[node.output[0]] = op
        self.register_port([node.input[0]], [node.output[0]], op)
      
      elif node.op_type == "Mul":
        if self.get_optype(i+1) != "Add": # lookahead
          raise NotImplementedError("No Add found after Mul, no valid implementation.")
        bias_name = self.nodes[i+1].input[1]
        
        is_relu = self.get_optype(i+2) == "Relu"
        op = MacOp(f"k{i:03d}mac", is_relu=is_relu)

        i += 1 # add
        if is_relu:
          print(f"WARNING: fusing Mul+Add+Relu")
          i += 1 # relu
        else:
          print(f"WARNING: fusing Mul+Add")
        
        onnx_out_name = self.nodes[i].output[0]
        op.register_params([self.get_tensor(tname) for tname in (*node.input, bias_name, onnx_out_name)])
        op.save_txt(self.data_path)
        self.onnxname_2_op[onnx_out_name] = op
        self.register_port([node.input[0]], [onnx_out_name], op)

      elif node.op_type == "Gemm":
        is_relu = self.get_optype(i+1) == "Relu" # lookahead
        op = GemmOp(f"k{i:03d}gemm", is_relu=is_relu)
        
        if self.get_optype(i+1) == "Relu":
          print(f"WARNING: fusing Gemm+Relu")
          i += 1
        
        onnx_out_name = self.nodes[i].output[0]
        op.register_params([self.get_tensor(tname) for tname in (*node.input, onnx_out_name)], 
                           node.attribute)
        op.save_txt(self.data_path)
        self.onnxname_2_op[onnx_out_name] = op
        self.register_port([node.input[0]], [onnx_out_name], op)
        
      elif node.op_type == "QuantizeLinear":
        op = QuantizeLinearOp(f"k{i:03d}quantizelinear")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)])
        op.save_txt(self.data_path)
        self.onnxname_2_op[node.output[0]] = op
        self.register_port([node.input[0]], [node.output[0]], op)

      elif node.op_type == "QLinearAdd": # reconvergent op
        is_relu = self.get_optype(i+1) == "Relu"
        op = QLinearAddOp(f"k{i:03d}qlinearadd", is_relu)
        
        if is_relu: # lookahead
          print(f"WARNING: fusing Add+Relu")
          i += 1
        
        onnx_out_name = self.nodes[i].output[0]
        op.register_params([self.get_tensor(tname) for tname in (*node.input, onnx_out_name)])
        op.save_txt(self.data_path)
        
        self.onnxname_2_op[onnx_out_name] = op
        self.register_port([node.input[0]], [onnx_out_name], op)
        
        # external caching
        buf_name = op.name + "_i_buf"
        self.gmiobuf_2_size[buf_name] = dtype_to_cstr(op.dtype), op.out_size
        skip_op = self.onnxname_2_op[node.input[3]]
        skip_op.attach_gmio_out(buf_name)
        self.adf_connects.append(skip_op.get_gmioout_connect_line())
        
        op.attach_gmio_in(buf_name)
        self.adf_connects.append(op.get_gmioin_connect_line(pin_idx=1))

      elif node.op_type == "QLinearConv":
        op = QLinearConvOp(f"k{i:03d}qlinearconv")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.onnxname_2_op[node.output[0]] = op
        self.register_port([node.input[0]], [node.output[0]], op)
      
      elif node.op_type == "QLinearGlobalAveragePool":
        op = QLinearPoolOp(f"k{i:03d}qlinearpool", reduction_mode="avg", is_global=True)
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.onnxname_2_op[node.output[0]] = op
        self.register_port([node.input[0]], [node.output[0]], op)

      elif node.op_type == "QLinearAveragePool":
        op = QLinearPoolOp(f"k{i:03d}qlinearpool", reduction_mode="avg", is_global=True)
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.onnxname_2_op[node.output[0]] = op
        self.register_port([node.input[0]], [node.output[0]], op)
        
      elif node.op_type == "QGemm":
        op = QGemmOp(f"k{i:03d}qgemm")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)],
                           node.attribute)
        op.save_txt(self.data_path)
        self.onnxname_2_op[node.output[0]] = op
        self.register_port([node.input[0]], [node.output[0]], op)
      
      elif node.op_type == "QLinearMul":
        if self.get_optype(i+1) != "QLinearAdd": # lookahead
          raise NotImplementedError("No QLinearAdd found after QLinearMul, no valid implementation.")
        add_node_inputs = self.nodes[i+1].input
        
        is_relu = self.get_optype(i+2) == "Relu"
        op = QLinearMacOp(f"k{i:03d}qlinearmac", is_relu=is_relu)

        i += 1 # add
        if is_relu:
          print(f"WARNING: fusing QLinearMul+QLinearAdd+Relu")
          i += 1 # relu
        else:
          print(f"WARNING: fusing QLinearMul+QLinearAdd")
        
        onnx_out_name = self.nodes[i].output[0]
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *add_node_inputs, onnx_out_name)])
        op.save_txt(self.data_path)
        self.onnxname_2_op[onnx_out_name] = op
        self.register_port([node.input[0]], [onnx_out_name], op)
      
      elif node.op_type == "DequantizeLinear":
        op = DequantizeLinearOp(f"k{i:03d}dequantizeLinear")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)])
        op.save_txt(self.data_path)
        self.onnxname_2_op[node.output[0]] = op
        self.register_port([node.input[0]], [node.output[0]], op)
      
      elif node.op_type == "MatMul":
        if self.get_optype(i+1) != "Add": # lookahead
          raise NotImplementedError("No Add found after MatMul, no valid implementation.")
        bias_name = self.nodes[i+1].input[1]
        
        is_relu = self.get_optype(i+2) == "Relu"
        op = GemmOp(f"k{i:03d}gemm", is_relu=is_relu)
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
        self.onnxname_2_op[onnx_out_name] = op
        self.register_port([node.input[0]], [onnx_out_name], op)
      
      elif node.op_type == "Softmax":
        op = SoftmaxOp(f"k{i:03d}softmax")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)])
        op.save_txt(self.data_path)
        self.onnxname_2_op[node.output[0]] = op
        self.register_port([node.input[0]], [node.output[0]], op)
      
      elif node.op_type == "QLinearSoftmax":
        op = QLinearSoftmaxOp(f"k{i:03d}qlinearsoftmax")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.onnxname_2_op[node.output[0]] = op
        self.register_port([node.input[0]], [node.output[0]], op)
      
      elif node.op_type == "Transpose":
        op = TransposeOp(f"k{i:03d}transpose")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.onnxname_2_op[node.output[0]] = op
        self.register_port([node.input[0]], [node.output[0]], op)
      
      else:
        raise ValueError(f"Unexpected op_type {node.op_type}")

      if self.is_output_all:
        for ioname in [*node.input, *node.output]:
          tensor = self.get_tensor(ioname)
          tensor_shapestr = "x".join(str(dim) for dim in tensor.shape)
          out_path = f"{i:03d}__{node.name}__{ioname}__{tensor_shapestr}.txt".replace("/", "_")
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
      if next(reversed(self.onnxname_2_op.values())).name == "k011qlinearadd":
        self.modelout_2_op = {"output": self.onnxname_2_op[next(reversed(self.onnxname_2_op))]}
        break
      
    self.onnxname_2_op[next(iter(self.onnxname_2_op))].disable_input_pad()
    self.onnxname_2_op[next(reversed(self.onnxname_2_op))].disable_output_pad()
