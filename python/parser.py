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


class ParsedGraph:

  def __init__(self, 
               name: str,
               data_count: int):
    self.name = name
    self.data_count = data_count

    self.onnxname_2_op: Mapping[str, OpParser] = OrderedDict()
    
    self.adf_connects: List[str] = []
    self.gmiobuf_2_size: Mapping[str, int] = {}

    self.in_ops: List[InputOp] = []
    self.out_ops: List[OpParser] = []
    self.optout_ops: List[OpParser] = []
    self.input_count = 0
    self.output_count = 0
    
  def register_port(self, 
                    onnx_innames: List[str], 
                    onnx_outnames: List[str], 
                    op: OpParser) -> None:
    for i, input_name in enumerate(onnx_innames):
      in_port = self.onnxname_2_op[input_name].get_adf_port_name()
      self.adf_connects.append(op.get_connect_line(in_port, i))
    
    gmio_connects = op.get_gmio_connect_line(len(onnx_innames))
    if gmio_connects != "":
      self.adf_connects.append(gmio_connects)

  def get_input_ops(self) -> List[OpParser]:
    input_ops = []
    for op in self.onnxname_2_op.values():
      if isinstance(op, InputOp):
        input_ops.append(op)
    return input_ops

  def get_first_op(self) -> OpParser:
    return next(iter(self.onnxname_2_op.values()))
  
  def get_last_op(self) -> OpParser:
    return next(reversed(self.onnxname_2_op.values()))

  def rename_last_op(self, new_key: str) -> None:
    last_op_key = next(reversed(self.onnxname_2_op))
    self.onnxname_2_op[new_key] = self.onnxname_2_op.pop(last_op_key)

  def register_metadata(self):
    i = 0 # index of plio
    for op in self.onnxname_2_op.values():
      if isinstance(op, InputOp):
        op.id = i
        i += 1
        self.in_ops.append(op)
        self.input_count += 1
    
    i = 0 # index of plout
    for op in self.onnxname_2_op.values():
      if op.is_output:
        op.id = i
        self.out_ops.append(op)
        i += 1
        self.output_count += 1
    for op in self.onnxname_2_op.values():
      if not op.is_output and not isinstance(op, InputOp):
        op.id = i
        i += 1
        self.optout_ops.append(op)

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
    
    model = onnx.load(onnx_path)
    self.graph = model.graph
    self.nodes = model.graph.node
    # for node in self.nodes:
    #   node_str = MessageToJson(node)
    #   with open("nodes.json", "a") as f:
    #     f.write(node_str)
    self.graph_name = os.path.splitext(os.path.basename(onnx_path))[0]
    
    self.register_initializers(model, input_tensors, output_tensors)

    # generate first subgraph, register model inputs and outputs
    self.graphs = [ParsedGraph(self.graph_name, data_count=data_count)]
    self.g = self.graphs[0]
    
    for i, (model_input, tensor) in enumerate(zip(model.graph.input, input_tensors)):
      self.g.onnxname_2_op[model_input.name] = InputOp(model_input.name, i, tensor)
      self.g.input_count += 1
    
    self.g.modelout_2_op = {i.name: None for i in model.graph.output}
  
  def register_initializers(self, model, input_tensors, output_tensors):
    self.tensor: Mapping[str, np.ndarray] = {
      init.name: numpy_helper.to_array(init)
      for init in model.graph.initializer
    }
    self.tensor.update({
      i.name: t 
      for i, t in zip(model.graph.input, input_tensors)
    })
    self.tensor.update(output_tensors)
  
  def get_optype(self, i: int):
    if i >= len(self.nodes):
      return ""
    return self.nodes[i].op_type
  
  def parse_filename(self,
                     filename: str, 
                     is_e2e: bool = True):
    filename = os.path.splitext(filename)[0]
    suffix = "_host" if is_e2e else ""

    fn_list = filename.split("shape")
    if len(fn_list) == 2 and is_e2e:
      shape_list = fn_list[1].split("x")
      shape_list[0] = f"{self.data_count}"
      filename = fn_list[0] + "shape" + "x".join(shape_list)
    
    return f"{filename}{suffix}.txt"

  # assumes nodes are in topological sorted order
  def parse(self):
    i = 0
    while i < len(self.nodes):
      node = self.nodes[i]

      if node.op_type in SKIPPABLE_NODES:
        last_op = self.g.get_last_op()
        node_output_name = node.output[0]
        if len(node_output_name) != 0 and np.all(self.tensor[node_output_name].flatten() == last_op.tout.flatten()):
          print(f"Found matching output {node_output_name} and {op.name} output")
          self.g.rename_last_op(new_key=node_output_name)
          last_op.disable_output_pad()
        else:
          print(f"WARNING: {node.op_type} not implemented, skipping...")
      
      elif node.op_type == "DequantizeLinear" and self.get_optype(i+1) in SKIPPABLE_NODES and self.get_optype(i+2) == "QuantizeLinear":
        last_op = self.g.get_last_op()
        node_output_name = self.nodes[i+2].output[0]
        if np.all(self.tensor[node_output_name].flatten() == last_op.tout.flatten()):
          print(f"Found matching output {node_output_name} and {op.name} output")
          self.g.rename_last_op(new_key=node_output_name)
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
        op.register_params([self.tensor[tname] for tname in (*node.input, onnx_out_name)])
        op.save_txt(self.data_path)
        self.g.onnxname_2_op[onnx_out_name] = op
        self.g.register_port(node.input, [onnx_out_name], op)
      
      elif node.op_type == "AveragePool":
        op = PoolOp(f"k{i:03d}pool", reduction_mode="avg")
        op.register_params([self.tensor[tname] for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.g.onnxname_2_op[onnx_out_name] = op
        self.g.register_port(node.input, [onnx_out_name], op)
      
      elif node.op_type == "Conv":
        is_relu = self.get_optype(i+1) == "Relu"
        op = ConvOp(f"k{i:03d}conv", is_relu)
        
        if is_relu: # lookahead
          print(f"WARNING: fusing Conv+Relu")
          i += 1
        
        onnx_out_name = self.nodes[i].output[0]
        op.register_params([self.tensor[tname] for tname in (*node.input, onnx_out_name)], node.attribute)
        op.save_txt(self.data_path)
        self.g.onnxname_2_op[onnx_out_name] = op
        self.g.register_port([node.input[0]], [onnx_out_name], op)
      
      elif node.op_type == "MaxPool":
        op = PoolOp(f"k{i:03d}pool", reduction_mode="max")
        op.register_params([self.tensor[tname] for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.g.onnxname_2_op[node.output[0]] = op
        self.g.register_port([node.input[0]], [node.output[0]], op)
      
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
        op.register_params([self.tensor[tname] for tname in (*node.input, bias_name, onnx_out_name)])
        op.save_txt(self.data_path)
        self.g.onnxname_2_op[onnx_out_name] = op
        self.g.register_port([node.input[0]], [onnx_out_name], op)

      elif node.op_type == "Gemm":
        is_relu = self.get_optype(i+1) == "Relu" # lookahead
        op = GemmOp(f"k{i:03d}gemm", is_relu=is_relu)
        
        if self.get_optype(i+1) == "Relu":
          print(f"WARNING: fusing Gemm+Relu")
          i += 1
        
        onnx_out_name = self.nodes[i].output[0]
        op.register_params([self.tensor[tname] for tname in (*node.input, onnx_out_name)], 
                           node.attribute)
        op.save_txt(self.data_path)
        self.g.onnxname_2_op[onnx_out_name] = op
        self.g.register_port([node.input[0]], [onnx_out_name], op)
        
      elif node.op_type == "QuantizeLinear":
        op = QuantizeLinearOp(f"k{i:03d}quantizelinear")
        op.register_params([self.tensor[tname] for tname in (*node.input, *node.output)])
        op.save_txt(self.data_path)
        self.g.onnxname_2_op[node.output[0]] = op
        self.g.register_port([node.input[0]], [node.output[0]], op)

      elif node.op_type == "QLinearAdd": # reconvergent op
        is_relu = self.get_optype(i+1) == "Relu"
        op = QLinearAddOp(f"k{i:03d}qlinearadd", is_relu)
        
        if is_relu: # lookahead
          print(f"WARNING: fusing Add+Relu")
          i += 1
        
        onnx_out_name = self.nodes[i].output[0]
        op.register_params([self.tensor[tname] for tname in (*node.input, onnx_out_name)])
        op.save_txt(self.data_path)
        
        self.g.onnxname_2_op[onnx_out_name] = op
        self.g.register_port([node.input[0]], [onnx_out_name], op)
        
        # external caching
        buf_name = op.name + "_i_buf"
        self.g.gmiobuf_2_size[buf_name] = dtype_to_cstr(op.dtype), op.out_size
        skip_op = self.g.onnxname_2_op[node.input[3]]
        skip_op.attach_gmio_out(buf_name)
        self.g.adf_connects.append(skip_op.get_gmioout_connect_line())
        
        op.attach_gmio_in(buf_name)
        self.g.adf_connects.append(op.get_gmioin_connect_line(pin_idx=1))

      elif node.op_type == "QLinearConv":
        op = QLinearConvOp(f"k{i:03d}qlinearconv")
        op.register_params([self.tensor[tname] for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.g.onnxname_2_op[node.output[0]] = op
        self.g.register_port([node.input[0]], [node.output[0]], op)
      
      elif node.op_type == "QLinearGlobalAveragePool":
        op = QLinearPoolOp(f"k{i:03d}qlinearpool", reduction_mode="avg", is_global=True)
        op.register_params([self.tensor[tname] for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.g.onnxname_2_op[node.output[0]] = op
        self.g.register_port([node.input[0]], [node.output[0]], op)

      elif node.op_type == "QLinearAveragePool":
        op = QLinearPoolOp(f"k{i:03d}qlinearpool", reduction_mode="avg", is_global=True)
        op.register_params([self.tensor[tname] for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.g.onnxname_2_op[node.output[0]] = op
        self.g.register_port([node.input[0]], [node.output[0]], op)
        
      elif node.op_type == "QGemm":
        op = QGemmOp(f"k{i:03d}qgemm")
        op.register_params([self.tensor[tname] for tname in (*node.input, *node.output)],
                           node.attribute)
        op.save_txt(self.data_path)
        self.g.onnxname_2_op[node.output[0]] = op
        self.g.register_port([node.input[0]], [node.output[0]], op)
      
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
        op.register_params([self.tensor[tname] for tname in (*node.input, *add_node_inputs, onnx_out_name)])
        op.save_txt(self.data_path)
        self.g.onnxname_2_op[onnx_out_name] = op
        self.g.register_port([node.input[0]], [onnx_out_name], op)
      
      elif node.op_type == "DequantizeLinear":
        op = DequantizeLinearOp(f"k{i:03d}dequantizeLinear")
        op.register_params([self.tensor[tname] for tname in (*node.input, *node.output)])
        op.save_txt(self.data_path)
        self.g.onnxname_2_op[node.output[0]] = op
        self.g.register_port([node.input[0]], [node.output[0]], op)
      
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
        op.register_params([self.tensor[tname] for tname in (*node.input, bias_name, onnx_out_name)],
                           node.attribute)
        op.save_txt(self.data_path)
        self.g.onnxname_2_op[onnx_out_name] = op
        self.g.register_port([node.input[0]], [onnx_out_name], op)
      
      elif node.op_type == "Softmax":
        op = SoftmaxOp(f"k{i:03d}softmax")
        op.register_params([self.tensor[tname] for tname in (*node.input, *node.output)])
        op.save_txt(self.data_path)
        self.g.onnxname_2_op[node.output[0]] = op
        self.g.register_port([node.input[0]], [node.output[0]], op)
      
      elif node.op_type == "QLinearSoftmax":
        op = QLinearSoftmaxOp(f"k{i:03d}qlinearsoftmax")
        op.register_params([self.tensor[tname] for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.g.onnxname_2_op[node.output[0]] = op
        self.g.register_port([node.input[0]], [node.output[0]], op)
      
      elif node.op_type == "Transpose":
        op = TransposeOp(f"k{i:03d}transpose")
        op.register_params([self.tensor[tname] for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.g.onnxname_2_op[node.output[0]] = op
        self.g.register_port([node.input[0]], [node.output[0]], op)
      
      else:
        raise ValueError(f"Unexpected op_type {node.op_type}")
      
      for output_name in self.nodes[i-1].output:
        if output_name in self.graph.output:
          op.is_output = True

      if self.is_output_all:
        for ioname in [*node.input, *node.output]:
          tensor = self.get_tensor[ioname]
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
      if next(reversed(self.g.onnxname_2_op.values())).name == "k011qlinearadd":
        op.is_output = True
        break
      
    self.g.get_first_op().disable_input_pad()
    self.g.get_last_op().disable_output_pad()

    self.g.register_metadata()
    
    for v in self.g.onnxname_2_op.values():
      print(v.name, "\t", v.get_computation_count()) 
