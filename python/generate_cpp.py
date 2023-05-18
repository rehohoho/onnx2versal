from typing import List, Mapping
from dataclasses import dataclass
import os

import numpy as np
import onnx
from onnx import numpy_helper

MAX_FLOAT_PARAMS = 16384//4


def dtype_to_cstr(np_dtype: np.dtype):
  if np_dtype == "float32":
    return "float"
  elif np_dtype == "int8":
    return "int"
  else:
    raise NotImplementedError(f"Not implemented type {np_dtype}")

class OpParser:
  include_file: str

  def __init__(self, name: str):
    self.name = name
  
  def get_include_line(self) -> str:
    return f'#include "{self.include_file}"'
  
  def get_arg_line(self) -> str:
    return f""
  
  def get_initlist_line(self) -> str:
    return f""
  
  def get_weight_line(self) -> str:
    return f""
  
  def get_callarg_line(self) -> str:
    return f""


class ArgmaxOp(OpParser):
  include_file: str = "graph_argmax.h"


class ConvOp(OpParser):
  include_file: str = "graph_conv.h"

  def register_params(self, 
                      tinput: np.ndarray, 
                      tweight: np.ndarray,
                      tbias: np.ndarray,
                      toutput: np.ndarray):
    inB, inC, inH, inW = tinput.shape
    wM, wC, wK1, wK2 = tweight.shape
    bM, = tbias.shape
    outB, outM, outH, outW = toutput.shape
    
    assert inH == inW and outH == outW and inC == wC and wM == bM and bM == outM \
      and inB == outB and wK1 == wK2

    self.INP_W = inH
    self.OUT_W = outH
    self.B = inB
    self.C = inC
    self.M = wM
    self.K = wK1
    self.dtype = tweight.dtype
    self.out_size = self.B * self.M * self.OUT_W * self.OUT_W
  
  def register_weights(self, 
                       weights: np.ndarray, 
                       bias: np.ndarray):
    """Expects MxCxKxK weights as per PyTorch
    Returns MxCxKxK' weights, with K' padded so K'%8=0
    """
    M, C, K, K = weights.shape
    k_pad = (8 - K%8) % 8
    if k_pad != 0:
      print(f"Padding Conv weights {M, C, K, K} to {M, C, K, K+k_pad}")
      weights = np.pad(weights, ((0,0),(0,0),(0,0),(0,k_pad)), "constant", constant_values=0)
    self.weights = weights
    self.bias = bias
  
  def get_kernel_line(self) -> str:
    graph = "ConvReluGraph"
    kernel = "ConvReluScalarBCHW"
    if self.OUT_W % 8 == 0 and self.K == 5:
      kernel = "Conv5x5on8ReluBCHW"
    
    return f"{graph}<{kernel},{self.INP_W},{self.OUT_W},{self.B},{self.C},{self.M},{self.K}> {self.name};"

  def get_arg_line(self) -> str:
    ctype = dtype_to_cstr(self.dtype)
    return f"std::vector<{ctype}> {self.name}_w,\nstd::vector<{ctype}> {self.name}_b"
  
  def get_initlist_line(self) -> str:
    return f"{self.name}({self.name}_w, {self.name}_b)"
  
  def get_weight_line(self) -> str:
    wstring = str(self.weights.flatten().tolist())[1:-1]
    bstring = str(self.bias.flatten().tolist())[1:-1]
    ctype = dtype_to_cstr(self.dtype)
    return f"std::vector<{ctype}> {self.name}_w {{{wstring}}};\n" + \
      f"std::vector<{ctype}> {self.name}_b {{{bstring}}};"
  
  def get_callarg_line(self) -> str:
    return f"{self.name}_w, {self.name}_b"
  

class GemmOp(OpParser):
  include_file: str = "graph_gemm.h"

  def register_params(self, 
                      tinput: np.ndarray, 
                      tweight: np.ndarray,
                      tbias: np.ndarray,
                      toutput: np.ndarray):
    inM, inK = tinput.shape
    wN, wK = tweight.shape
    bN, = tbias.shape
    outM, outN = toutput.shape
    
    assert inM == outM and inK == wK and wN == bN and bN == outN

    self.M = inM
    self.K = inK
    self.N = wN
    self.dtype = tweight.dtype
    self.out_size = self.M*self.N
  
  def register_weights(self, 
                       weights: np.ndarray,
                       bias: np.ndarray):
    """Expects NxK weights as per PyTorch
    Returns KxN weights, with N padded so N%4=0
    """
    weights = weights.transpose(1,0)
    K, N = weights.shape
    n_pad = (4 - N%4) % 4
    if n_pad != 0:
      print(f"Padding Gemm weights {K, N} to {K, N+n_pad}")
      weights = np.pad(weights, ((0,0),(0,n_pad)), "constant", constant_values=0)
    self.weights = weights
    self.bias = bias
  
  def get_kernel_type(self) -> str:
    graph = "GemmReluMkknChunkGraph"
    kernel = "GemmReluScalarMKKN"
    concat_kernel = "ConcatScalar"
    chunkSize = int(MAX_FLOAT_PARAMS/self.K//4*4)
    if self.K % 2 == 0 and chunkSize % 4 == 0:
      kernel = "GemmReluMKKN"
    if chunkSize % 8 == 0 and self.N % 4 == 0:
      concat_kernel = "ConcatVector"
    
    return f"{graph}<{kernel},{concat_kernel},{chunkSize},{self.M},{self.K},{self.N}>"
  
  def get_kernel_line(self) -> str:
    return f"{self.get_kernel_type()} {self.name};"

  def get_arg_line(self) -> str:
    ctype = dtype_to_cstr(self.dtype)
    return f"std::vector<{ctype}> {self.name}_w,\nstd::vector<{ctype}> {self.name}_b"
  
  def get_initlist_line(self) -> str:
    return f"{self.name}({self.name}_w, {self.name}_b)"
  
  def get_weight_line(self) -> str:
    wstring = str(self.weights.flatten().tolist())[1:-1]
    bstring = str(self.bias.flatten().tolist())[1:-1]
    ctype = dtype_to_cstr(self.dtype)
    return f"std::vector<{ctype}> {self.name}_w {{{wstring}}};\n" + \
      f"std::vector<{ctype}> {self.name}_b {{{bstring}}};"
  
  def get_callarg_line(self) -> str:
    return f"{self.name}_w, {self.name}_b"


class PoolOp(OpParser):
  include_file: str = "graph_pool.h"
  
  def register_params(self, 
                      tinput: np.ndarray, 
                      toutput: np.ndarray):
    inB, inC, inH, inW = tinput.shape
    outB, outC, outH, outW = toutput.shape
    
    assert inH == inW and outH == outW and inC == outC and inB == outB

    self.INP_W = inH
    self.OUT_W = outH
    self.B = inB
    self.C = inC
    self.dtype = toutput.dtype
    self.out_size = self.B * self.C * self.OUT_W * self.OUT_W
  
  def get_kernel_line(self) -> str:
    graph = "MaxpoolGraph"
    kernel = "MaxpoolScalarBCHW"
    if self.OUT_W % 4 == 0:
      kernel = "Maxpool2x2BCHW"
    return f"{graph}<{kernel},{self.INP_W},{self.OUT_W},{self.B},{self.C}> {self.name};"


class CppGenerator:

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
    self.graph_name = os.path.splitext(os.path.basename(onnx_path))[0]
    
    # register model input, node output and model outputs
    self.modelin_2_tensor = {i.name: t for i, t in zip(model.graph.input, input_tensors)}
    for i, model_input in enumerate(model.graph.input):
      self.nodeout_2_adfport[model_input.name] = f"plin[{i}].out[0]"
    self.modelout_2_op = {i.name: None for i in model.graph.output}

    # save inputs
    for input_name, input_tensor in self.modelin_2_tensor.items():
      np.savetxt(f"{self.data_path}/{input_name}.txt", input_tensor.reshape(-1, 2))

    # store I/O tensors and model parameters
    self.input_tensors: List[np.ndarray] = input_tensors
    self.initializers: Mapping[str, np.ndarray] = {
      init.name: init for init in model.graph.initializer}
    self.output_tensors: Mapping[str, np.ndarray] = output_tensors
  
  def get_tensor(self, name: str):
    if name in self.modelin_2_tensor:
      return self.modelin_2_tensor[name]
    elif name in self.initializers:
      return numpy_helper.to_array(self.initializers[name])
    elif name in self.output_tensors:
      return self.output_tensors[name]
    else:
      raise ValueError(f"Unable to find {name} in initializers or output_tensors.")
    
  # assumes nodes are in topological sorted order
  def parse(self):
    for i, node in enumerate(self.nodes):
      
      if node.op_type == "Conv":
        if self.nodes[i+1].op_type != "Relu": # lookahead
          raise NotImplementedError("No relu found after Conv, no valid implementation.")
        print(f"WARNING: fusing Conv+Relu")

        onnx_out_name = self.nodes[i+1].output[0]
        tinput = self.get_tensor(node.input[0])
        tweight = self.get_tensor(node.input[1])
        tbias = self.get_tensor(node.input[2])
        toutput = self.get_tensor(onnx_out_name)
        
        op = ConvOp(f"k{i}conv")
        op.register_params(tinput, tweight, tbias, toutput)
        op.register_weights(tweight, tbias)
        self.op_list.append(op)

        self.adf_connects.append(
          f"adf::connect<> ({self.nodeout_2_adfport[node.input[0]]}, {op.name}.pin[0]);")
        self.nodeout_2_adfport[onnx_out_name] = f"{op.name}.pout[0]"
        if onnx_out_name in self.modelout_2_op:
          self.modelout_2_op[onnx_out_name] = op
        np.savetxt(f"{self.data_path}/{op.name}_in.txt", tinput.reshape(-1, 2))
        np.savetxt(f"{self.data_path}/{op.name}_goldenout.txt", toutput.reshape(-1, 2))
      
      elif node.op_type == "Relu":
        # handled by fusing with previous
        pass
            
      elif node.op_type == "MaxPool":
        onnx_out_name = node.output[0]
        tinput = self.get_tensor(node.input[0])
        toutput = self.get_tensor(onnx_out_name)
        
        op = PoolOp(f"k{i}pool")
        op.register_params(tinput, toutput)
        self.op_list.append(op)

        self.adf_connects.append(
          f"adf::connect<> ({self.nodeout_2_adfport[node.input[0]]}, {op.name}.pin[0]);")
        self.nodeout_2_adfport[onnx_out_name] = f"{op.name}.pout[0]"
        if onnx_out_name in self.modelout_2_op:
          self.modelout_2_op[onnx_out_name] = op
        np.savetxt(f"{self.data_path}/{op.name}_in.txt", tinput.reshape(-1, 2))
        np.savetxt(f"{self.data_path}/{op.name}_goldenout.txt", toutput.reshape(-1, 2))

      elif node.op_type == "Gemm":
        if self.nodes[i+1].op_type != "Relu": # lookahead
          raise NotImplementedError("No relu found after Gemm, no valid implementation.")
        
        onnx_out_name = self.nodes[i+1].output[0]
        tinput = self.get_tensor(node.input[0])
        tweight = self.get_tensor(node.input[1])
        tbias = self.get_tensor(node.input[2])
        toutput = self.get_tensor(onnx_out_name)

        op = GemmOp(f"k{i}gemm")
        op.register_params(tinput, tweight, tbias, toutput)
        op.register_weights(tweight, tbias)
        self.op_list.append(op)

        self.adf_connects.append(
          f"for (int i = 0; i < {op.get_kernel_type()}::CHUNK_COUNT; i++)\n" + \
          f"  adf::connect<> ({self.nodeout_2_adfport[node.input[0]]}, {op.name}.pin[i]);")
        self.nodeout_2_adfport[onnx_out_name] = f"{op.name}.pout[0]"
        if onnx_out_name in self.modelout_2_op:
          self.modelout_2_op[onnx_out_name] = op
        np.savetxt(f"{self.data_path}/{op.name}_in.txt", tinput.reshape(-1, 2))
        np.savetxt(f"{self.data_path}/{op.name}_goldenout.txt", toutput.reshape(-1, 2))
        
      elif node.op_type in ["Shape", "Constant", "Gather", "Unsqueeze", "Concat", "Reshape"]:
        print(f"WARNING: {node.op_type} not implemented, skipping...")
        if len(node.output[0]) != 0 and \
          np.all(self.get_tensor(node.output[0]).flatten() == toutput.flatten()):
          print(f"Found matching output {node.output[0]} and {op.name} output")
          self.nodeout_2_adfport[node.output[0]] = f"{op.name}.pout[0]"
      
      else:
        raise ValueError(f"Unexpected op_type {node.op_type}")

      if self.is_output_all:
        for ioname in [*node.input, *node.output]:
          tensor = self.get_tensor(ioname)
          tensor_shapestr = "x".join(str(dim) for dim in tensor.shape)
          out_path = f"{i}__{node.name}__{ioname}__{tensor_shapestr}.txt".replace("/", "_")
          if "weight" in ioname or "bias" in ioname:
            out_name = ioname.replace("/", "_").replace(".", "_")
            tensor = str(tensor.flatten().tolist())[1:-2]
            tmp = f"std::vector<{dtype_to_cstr(tensor.dtype)}> {out_name} {{{tensor}}};"
            with open(out_path, "w") as f: 
              f.write(tmp)
          else:
            if tensor.size > 2: 
              tensor = tensor.reshape(-1, 2)
            else: 
              tensor = tensor.reshape(-1)
            np.savetxt(out_path, tensor)

  def get_includes(self) -> str:
    include_list = set(i.get_include_line() for i in self.op_list)
    return "\n".join(include_list)
  
  def get_kernels(self) -> str:
    return "    " + "\n".join(i.get_kernel_line() for i in self.op_list).replace("\n", "\n    ")

  def get_args(self) -> str:
    args = [f"const std::string& {inpname}" for inpname in self.modelin_2_tensor]
    args += [f"const std::string& {op.name}_out" for op in self.modelout_2_op.values()]
    args += [i.get_arg_line() for i in self.op_list]
    args += [f"const std::string& {op.name}_out = std::string()" for op in self.op_list 
             if op not in self.modelout_2_op.values()]
    args = [i for i in args if i != ""]
    return "      " + ",\n".join(args).replace("\n", "\n      ")
  
  def get_initlist(self) -> str:
    initlists = [i.get_initlist_line() for i in self.op_list]
    initlists = [i for i in initlists if i != ""]
    return "      " + ",\n".join(initlists).replace("\n", "\n      ")
  
  def get_plins(self) -> str:
    plins = [
      f'plin[{i}] = adf::input_plio::create("plin{i}_"+id+"_{inpname}", PLIO64_ARG({inpname}));'
      for i, inpname in enumerate(self.modelin_2_tensor)
    ]
    return "      " + "\n".join(plins).replace("\n", "\n      ")
  
  def get_plouts(self) -> str:
    plouts = [
      f'adf::output_plio a = adf::output_plio::create("plout0_"+id+"_{op.name}", PLIO64_ARG({op.name}_out));\n' + \
      f"plout.push_back(a);\n" + \
      f"adf::connect<> ({op.name}.pout[0], a.in[0]);"
      for op in self.modelout_2_op.values()
    ]
    return "      " + "\n".join(plouts).replace("\n", "\n      ")
  
  def get_optional_plouts(self) -> str:
    optplouts = [
      f'SET_OPT_PLOUT({op.name}_out, adf::connect<> ({op.name}.pout[0], a.in[0]), "{op.name}");'
      for op in self.op_list if op not in self.modelout_2_op.values()]
    return "      " + "\n".join(optplouts).replace("\n", "\n      ")
  
  def get_interkernel_connects(self) -> str:
    return "      " + "\n".join(self.adf_connects).replace("\n", "\n      ")
  
  def get_weights(self) -> str:
    weights = [i.get_weight_line() for i in self.op_list]
    weights = [i for i in weights if i != ""]
    return "\n".join(weights)
  
  def get_callargs(self, is_output_inter: bool) -> str:
    args = ['"input.txt"']
    args += [f'"{op.name}_goldenout.txt"' for op in self.modelout_2_op.values()]
    args += [i.get_callarg_line() for i in self.op_list]
    if is_output_inter:
      args += [f'"{op.name}_goldenout.txt"' for op in self.op_list 
               if op not in self.modelout_2_op.values()]
    args = [i for i in args if i != ""]
    return "  " + ",\n".join(args).replace("\n", "\n  ")
  
  def generate_cpp_graph_str(self):
    return f""" 
#include <adf.h>
{self.get_includes()}
#include "graph_utils.h"


class {self.graph_name.capitalize()} : public adf::graph {{

  private:
{self.get_kernels()}

  public:
    adf::input_plio plin[{len(self.modelin_2_tensor)}];
    std::vector<adf::output_plio> plout; // intermediate outputs optional

    {self.graph_name.capitalize()}(
      const std::string& id,
{self.get_args()}
    ): 
{self.get_initlist()}
    {{ 
      // mandatory input
{self.get_plins()}

      // mandatory output
{self.get_plouts()}

#define SET_OPT_PLOUT(TXT_PATH, STMT, PLOUT_NAME) \\
      if (!TXT_PATH.empty()) {{ \\
        std::string plout_name = "plout"+std::to_string(plout.size())+"_"+id+"_"+PLOUT_NAME; \\
        adf::output_plio a = adf::output_plio::create(plout_name, PLIO64_ARG(TXT_PATH)); \\
        STMT; plout.push_back(a);}} 

      // optional output
{self.get_optional_plouts()}

      // interkernel
{self.get_interkernel_connects()}
      
    }}
}};

{self.get_weights()}

// Unable to map 8 or more outputs on hardware since <= 8 cascade lines
#ifdef __OUTPUT_INTER__
{self.graph_name.capitalize()} {self.graph_name} (
  "{self.graph_name}",
{self.get_callargs(is_output_inter=True)}
);
#else
{self.graph_name.capitalize()} {self.graph_name} (
  "{self.graph_name}",
{self.get_callargs(is_output_inter=False)}
);
#endif


#ifdef __X86SIM__
int main(int argc, char ** argv) {{
	adfCheck({self.graph_name}.init(), "init {self.graph_name}");
  adfCheck({self.graph_name}.run(ITER_CNT), "run {self.graph_name}");
	adfCheck({self.graph_name}.end(), "end {self.graph_name}");
  return 0;
}}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {{
	adfCheck({self.graph_name}.init(), "init {self.graph_name}");
  get_graph_throughput_by_port({self.graph_name}, "plout[0]", {self.graph_name}.plout[0], 1*10, sizeof(float), ITER_CNT);
	adfCheck({self.graph_name}.end(), "end {self.graph_name}");
  return 0;
}}
#endif
"""
  
  def generate_cpp_graph(self):
    with open(f"../design/aie_src/graph_{self.graph_name}.cpp", "w") as f:
      f.write(self.generate_cpp_graph_str())
  
  def get_xtg_masters(self, is_output_inter: bool) -> str:
    masters = [
      f'("plin{i}_{self.graph_name}_{inpname}", f"{{args.input_dir}}/{inpname}.txt", 64, ' + \
      f'"{str(self.modelin_2_tensor[inpname].dtype)}")'
      for i, inpname in enumerate(self.modelin_2_tensor)
    ]
    if not is_output_inter:
      masters = [i.replace(".txt", "_host.txt") for i in masters]
    return "    " + ",\n".join(masters).replace("\n", "\n    ")
  
  def get_xtg_slaves(self, is_output_inter: bool) -> str:
    slaves = []
    for i, op in enumerate(self.modelout_2_op.values()):
      size = op.out_size
      file_suffix = "_goldenout.txt"
      if not is_output_inter:
        size *= self.data_count
        file_suffix = file_suffix.replace(".txt", "_host.txt")
      slaves += [
        f'("plout{i}_{self.graph_name}_{op.name}", f"{{args.output_dir}}/{op.name}{file_suffix}", ' + \
        f'64, "{str(op.dtype)}", {size})'
        for i, op in enumerate(self.modelout_2_op.values())
      ]

    if is_output_inter:
      i = len(self.modelout_2_op)
      for op in self.op_list:
        if op in self.modelout_2_op.values(): continue
        slaves.append(
          f'("plout{i}_{self.graph_name}_{op.name}", f"{{args.output_dir}}/{op.name}_goldenout.txt", 64, "{str(op.dtype)}", {op.out_size})')
        i += 1
    return "    " + ",\n".join(slaves).replace("\n", "\n    ")
  
  def generate_xtg_python_str(self, is_output_inter: bool):
    return f"""
import argparse
import logging

from xtg_aie import ExternalTraffic

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', required=True)
  parser.add_argument('--output_dir', required=True)
  args = parser.parse_args()
  
  logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")

  master_list = [
{self.get_xtg_masters(is_output_inter=is_output_inter)}
  ]

  slave_list = [
{self.get_xtg_slaves(is_output_inter=is_output_inter)}
  ]
  
  design = ExternalTraffic(master_list, slave_list)
  design.run()
"""

  def generate_xtg_python(self):
    with open(f"../design/trafficgen/xtg_{self.graph_name}.py", "w") as f:
      f.write(self.generate_xtg_python_str(is_output_inter=False))
    with open(f"../design/trafficgen/xtg_{self.graph_name}_output_inter.py", "w") as f:
      f.write(self.generate_xtg_python_str(is_output_inter=True))
  
  def get_cfg_input_kernels(self) -> str:
    tmp = f"nk=mm2s:{len(self.modelin_2_tensor)}:"
    for i in range(len(self.modelin_2_tensor)):
      tmp += f"mm2s_{i},"
    return tmp.strip(",")
  
  def get_cfg_output_kernels(self, is_output_inter: bool) -> str:
    out_cnt = len(self.modelout_2_op)
    if is_output_inter:
      out_cnt = len(self.op_list)
    
    tmp = f"nk=s2mm:{out_cnt}:"
    for i in range(out_cnt):
      tmp += f"s2mm_{i},"
    return tmp.strip(",")
  
  def get_cfg_input_scs(self) -> str:
    in_scs = [
      f"stream_connect=mm2s_{i}.s:ai_engine_0.plin{i}_{self.graph_name}_{inpname}"
      for i, inpname in enumerate(self.modelin_2_tensor)
    ]
    return "\n".join(in_scs)

  def get_cfg_output_scs(self, is_output_inter: bool) -> str:
    out_scs = [
      f"stream_connect=ai_engine_0.plout{i}_{self.graph_name}_{op.name}:s2mm_{i}.s"
      for i, op in enumerate(self.modelout_2_op.values())
    ]
    if is_output_inter:
      i = len(self.modelout_2_op)
      for op in self.op_list:
        if op in self.modelout_2_op.values(): continue
        out_scs.append(f"stream_connect=ai_engine_0.plout{i}_{self.graph_name}_{op.name}:s2mm_{i}.s")
        i += 1
    return "\n".join(out_scs)  
  
  def generate_cfg_str(self, is_output_inter: bool):
    return f"""
[connectivity]
{self.get_cfg_input_kernels()}
{self.get_cfg_output_kernels(is_output_inter=is_output_inter)}

#Connections For LeNET Insts 0...
{self.get_cfg_input_scs()}
{self.get_cfg_output_scs(is_output_inter=is_output_inter)}

[advanced]
# Disable Profiling in hw_emu so that it is faster...
param=hw_emu.enableProfiling=false
"""

  def generate_cfg(self):
    with open(f"../design/system_configs/{self.graph_name}.cfg", "w") as f:
      f.write(self.generate_cfg_str(is_output_inter=False))
    with open(f"../design/system_configs/{self.graph_name}_output_inter.cfg", "w") as f:
      f.write(self.generate_cfg_str(is_output_inter=True))

  def get_host_datafiles(self) -> str:
    outfiles = [
      f'#define INPUT{i}_FILENAME "{inpname}.txt"'
      for i, inpname in enumerate(self.modelin_2_tensor)
    ]
    outfiles += [
      f'#define OUTPUT{i}_FILENAME "{op.name}_goldenout.txt"'
      for i, op in enumerate(self.modelout_2_op.values())
    ]
    outfiles_host = [i.replace(".txt", "_host.txt") for i in outfiles]
    outfiles = ["#ifdef __OUTPUT_INTER__"] + outfiles + ["#else"] + outfiles_host + ["#endif"]
    
    n_outs = len(self.modelout_2_op)
    outfiles += [
      f'#define INTER{n_outs+i}_FILENAME "{op.name}_goldenout.txt"'
      for i, op in enumerate(self.op_list) if op not in self.modelout_2_op.values()
    ]
    return "\n".join(outfiles)
  
  def get_host_input_inits(self) -> str:
    inp_inits = []
    inp_initsyncs = ["", "#ifdef __IS_SW_EMU__"]
    
    for i, input_name in enumerate(self.modelin_2_tensor):
      input_tensor = self.modelin_2_tensor[input_name]
      size = input_tensor.size
      dtype = dtype_to_cstr(input_tensor.dtype)
      inp_inits.append(f"""xrtBufferHandle in{i}_bohdl = xrtBOAlloc(dhdl, iter_cnt*{size}*sizeof({dtype}), 0, 0);
auto in{i}_bomapped = reinterpret_cast<{dtype}*>(xrtBOMap(in{i}_bohdl));
printf("Input{i} memory virtual addr 0x%p\\n", in{i}_bomapped);

std::ifstream inp_file;
inp_file.open(data_dir+INPUT{i}_FILENAME, std::ifstream::in);
if (!inp_file) printf("Unable to open %s.\\n", (data_dir+INPUT{i}_FILENAME).c_str());
{dtype} d;
for (int j = 0; j < iter_cnt*{size}; j+=V_PER_LINE) {{
  for (int k = 0; k < V_PER_LINE; k++) {{
      inp_file >> d;
      in{i}_bomapped[j+k] = d;
  }}
}}
xrtKernelHandle in{i}_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "mm2s:{{mm2s_{i}}}");
xrtRunHandle in{i}_rhdl = xrtRunOpen(in{i}_khdl); 
xrtRunSetArg(in{i}_rhdl, 0, in{i}_bohdl);
xrtRunSetArg(in{i}_rhdl, 2, iter_cnt*{size});
xrtRunStart(in{i}_rhdl);""")
      inp_initsyncs.append(
        f"xrtBOSync(in{i}_bohdl, XCL_BO_SYNC_BO_TO_DEVICE, iter_cnt*{size}*sizeof({dtype}), 0);"
      )
    
    inp_initsyncs.append("#endif")
    return "   " + "\n".join(inp_inits+inp_initsyncs).replace("\n", "\n   ")
  
  def get_host_input_closes(self) -> str:
    inp_closes = [f"""auto in{i}_state = xrtRunWait(in{i}_rhdl);    
printf("mm2s completed with status (%d)\\n", in{i}_state);
xrtRunClose(in{i}_rhdl);
xrtKernelClose(in{i}_khdl);
xrtBOFree(in{i}_bohdl);"""
      for i in range(len(self.modelin_2_tensor))
    ]
    return "   " + "\n".join(inp_closes).replace("\n", "\n   ")
  
  def get_host_output_inits(self) -> str:
    out_inits = []
    for i, op in enumerate(self.modelout_2_op.values()):
      dtype = dtype_to_cstr(op.dtype)
      out_inits.append(f"""xrtBufferHandle out{i}_bohdl = xrtBOAlloc(dhdl, iter_cnt*{op.out_size}*sizeof({dtype}), 0, 0);
auto out{i}_bomapped = reinterpret_cast<{dtype}*>(xrtBOMap(out{i}_bohdl));
printf("Output{i} memory virtual addr 0x%p\\n", out{i}_bomapped);

xrtKernelHandle out{i}_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "s2mm:{{s2mm_{i}}}");
xrtRunHandle out{i}_rhdl = xrtRunOpen(out{i}_khdl); 
xrtRunSetArg(out{i}_rhdl, 0, out{i}_bohdl);
xrtRunSetArg(out{i}_rhdl, 2, iter_cnt*{op.out_size});
xrtRunStart(out{i}_rhdl);""")
    return "   " + "\n".join(out_inits).replace("\n", "\n   ")

  def get_host_output_closes(self) -> str:
    out_closes = [
    f"""auto out{i}_state = xrtRunWait(out{i}_rhdl);
printf("s2mm completed with status (%d)\\n", out{i}_state);
xrtRunClose(out{i}_rhdl);
xrtKernelClose(out{i}_khdl);
#ifdef __IS_SW_EMU__
xrtBOSync(out{i}_bohdl, XCL_BO_SYNC_BO_FROM_DEVICE, iter_cnt*{op.out_size}*sizeof({dtype_to_cstr(op.dtype)}), 0);
#endif
write_arr_to_file(out_dir+OUTPUT{i}_FILENAME, out{i}_bomapped, iter_cnt*{op.out_size});
xrtBOFree(out{i}_bohdl);"""
      for i, op in enumerate(self.modelout_2_op.values())
    ]
    return "   " + "\n".join(out_closes).replace("\n", "\n   ")

  def get_host_optout_inits(self) -> str:
    optout_inits = []
    i = len(self.modelout_2_op)
    for op in self.op_list:
      if op in self.modelout_2_op.values(): continue
      dtype = dtype_to_cstr(op.dtype)
      optout_inits.append(f"""
xrtBufferHandle inter{i}_bohdl = xrtBOAlloc(dhdl, iter_cnt*{op.out_size}*sizeof({dtype}), 0, 0);
auto inter{i}_bomapped = reinterpret_cast<{dtype}*>(xrtBOMap(inter{i}_bohdl));
printf("Inter{i} memory virtual addr 0x%p\\n", inter{i}_bomapped);
xrtKernelHandle inter{i}_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "s2mm:{{s2mm_{i}}}");
xrtRunHandle inter{i}_rhdl = xrtRunOpen(inter{i}_khdl);
xrtRunSetArg(inter{i}_rhdl, 0, inter{i}_bohdl);
xrtRunSetArg(inter{i}_rhdl, 2, iter_cnt*{op.out_size});
xrtRunStart(inter{i}_rhdl);""")
      i += 1
    return "   " + "\n".join(optout_inits).replace("\n", "\n   ")
  
  def get_host_optout_closes(self) -> str:
    optout_closes = []
    optout_syncs = ["#ifdef __IS_SW_EMU__"]
    optout_writes = []
    i = len(self.modelout_2_op)
    for op in self.op_list:
      if op in self.modelout_2_op.values(): continue
      optout_closes.append(f"""auto inter{i}_state = xrtRunWait(inter{i}_rhdl);
printf("inter{i} completed with status (%d)\\n", inter{i}_state);
xrtRunClose(inter{i}_rhdl);
xrtKernelClose(inter{i}_khdl);""")
      optout_syncs.append(
        f"xrtBOSync(inter{i}_bohdl, XCL_BO_SYNC_BO_FROM_DEVICE, iter_cnt*{op.out_size}*sizeof({dtype_to_cstr(op.dtype)}), 0);")
      optout_writes.append(f"""write_arr_to_file(out_dir+INTER{i}_FILENAME, inter{i}_bomapped, iter_cnt*{op.out_size});
xrtBOFree(inter{i}_bohdl);""")
      i += 1
    optout_syncs.append("#endif")
    return "   " + "\n".join(optout_closes + optout_syncs + optout_writes).replace("\n", "\n   ")

  def generate_host_cpp_str(self) -> str:
    return f"""
#include <fstream>

#include "graph_{self.graph_name}.cpp"

#include "experimental/xrt_aie.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_bo.h"
#include "experimental/xrt_error.h"

#include "adf/adf_api/XRTConfig.h"


#define V_PER_LINE      2
{self.get_host_datafiles()}

void write_arr_to_file(
   const std::string& filename,
   const float* bomapped,
   const size_t bosize
) {{
   std::ofstream file;
   file.open(filename, std::ofstream::out);
   if (!file) printf("Unable to open %s\\n", filename.c_str());
   for (int j = 0; j < bosize; j+=V_PER_LINE) {{
      for (int k = 0; k < V_PER_LINE; k++) {{
         file << bomapped[j+k] << " ";
      }}
      file << std::endl;
   }}
}}


static std::vector<char> load_xclbin(
   xrtDeviceHandle device, 
   const std::string &fnm
) {{
   if (fnm.empty())
      throw std::runtime_error("No xclbin specified");
   
   // load bit stream
   std::ifstream stream(fnm);
   stream.seekg(0,stream.end);
   size_t size = stream.tellg();
   stream.seekg(0,stream.beg);
   
   std::vector<char> header(size);
   stream.read(header.data(),size);
   
   auto top = reinterpret_cast<const axlf*>(header.data());
   if (xrtDeviceLoadXclbin(device, top))
      throw std::runtime_error("Bitstream download failed");
   
   return header;
}}

int main(int argc, char ** argv) {{
   // Parse args
   if(argc != 5) {{
      std::cout << "Usage: " << argv[0] <<" <xclbin>" << " <iter_cnt>" << " <data_dir>" << " <out_dir>" << std::endl;
      return EXIT_FAILURE;
   }}
   const char* xclbin_path = argv[1];
   const int iter_cnt = atoi(argv[2]);
   std::string data_dir = argv[3];
   data_dir.append("/");
   std::string out_dir = argv[4];
   out_dir.append("/");
   printf("\\nConfig:\\nxclbin: %s\\niter_cnt: %d\\ndata_dir: %s\\nout_dir: %s\\n\\n", 
      xclbin_path, iter_cnt, data_dir.c_str(), out_dir.c_str());

   // Open device, load xclbin
   auto deviceIdx = xrt::device(0);
   auto dhdl = xrtDeviceOpen(0);
   auto xclbin = load_xclbin(dhdl, xclbin_path);
   auto top = reinterpret_cast<const axlf*>(xclbin.data());


   // Allocate BOs (buffer objects) of requested size with appropriate flags
   // Memory map BOs into user's address space (DDR Memory)
   // Create kernel handle, runtime handle, set args, start kernels

   // Inputs
{self.get_host_input_inits()}
  
   // Outputs
{self.get_host_output_inits()}

#ifdef __OUTPUT_INTER__
{self.get_host_optout_inits()}
#endif


   // Graph execution for AIE
   adf::registerXRT(dhdl, top->m_header.uuid);
   try {{
      adfCheck({self.graph_name}.init(), "init {self.graph_name}");
#ifdef __IS_SW_EMU__
      adfCheck({self.graph_name}.run(iter_cnt), "run {self.graph_name}");
      adfCheck({self.graph_name}.wait(), "wait {self.graph_name}");
#else
      get_graph_throughput_by_port({self.graph_name}, "plout[0]", {self.graph_name}.plout[0], 1*iter_cnt, sizeof(float_t), iter_cnt);
#endif
      adfCheck({self.graph_name}.end(), "end {self.graph_name}");
   }}
   catch (const std::system_error& ex) {{
      xrt::error error(deviceIdx, XRT_ERROR_CLASS_AIE);
      auto errCode = error.get_error_code();
      auto timestamp = error.get_timestamp();
      auto err_str = error.to_string();
      std::cout << timestamp << " error code:" << errCode << " Error:" << err_str << std::endl;
   }}

   
   // Wait for Kernel execution to end, close runtime and kernel handlers
   printf("Waiting for dma hls to complete...\\n");
   
   // Close input handlers
{self.get_host_input_closes()}
   
   // Close output handlers
{self.get_host_output_closes()}

   printf("Closed runtime handlers and kernel handlers...\\n");

#ifdef __OUTPUT_INTER__
{self.get_host_optout_closes()}
#endif

   xrtDeviceClose(dhdl);
   return 0;
}}
"""

  def generate_host_cpp(self) -> str:
    with open(f"../design/host_app_src/{self.graph_name}_aie_app.cpp", "w") as f:
      f.write(self.generate_host_cpp_str())