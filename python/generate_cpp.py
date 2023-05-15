from typing import List, Mapping
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import numpy_helper


class OpParser:
  include_file: str

  def __init__(self, name: str):
    self.name = name

  def save_output(self, output: np.ndarray):
    np.savetxt(f"../data/{self.name}_goldenout.txt", output.reshape(-1, 2))
  
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
    return f"ConvReluGraph<Conv5x5on8ReluBCHW,{self.INP_W},{self.OUT_W},{self.B},{self.C},{self.M},{self.K}> {self.name};"

  def get_arg_line(self) -> str:
    return f"std::vector<float> {self.name}_w,\nstd::vector<float> {self.name}_b"
  
  def get_initlist_line(self) -> str:
    return f"{self.name}({self.name}_w, {self.name}_b)"
  
  def get_weight_line(self) -> str:
    wstring = str(self.weights.flatten().tolist())[1:-1]
    bstring = str(self.bias.flatten().tolist())[1:-1]
    return f"std::vector<float> {self.name}_w {{{wstring}}};\n" + \
      f"std::vector<float> {self.name}_b {{{bstring}}};"
  
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
    return f"GemmReluMkknChunkGraph<GemmReluMKKN,ConcatVector,MAX_FLOAT_PARAMS/{self.K}/4*4,{self.M},{self.K},{self.N}>"
  
  def get_kernel_line(self) -> str:
    return f"{self.get_kernel_type()} {self.name};"

  def get_arg_line(self) -> str:
    return f"std::vector<float> {self.name}_w,\nstd::vector<float> {self.name}_b"
  
  def get_initlist_line(self) -> str:
    return f"{self.name}({self.name}_w, {self.name}_b)"
  
  def get_weight_line(self) -> str:
    wstring = str(self.weights.flatten().tolist())[1:-1]
    bstring = str(self.bias.flatten().tolist())[1:-1]
    return f"std::vector<float> {self.name}_w {{{wstring}}};\n" + \
      f"std::vector<float> {self.name}_b {{{bstring}}};"
  
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
  
  def get_kernel_line(self) -> str:
    return f"MaxpoolGraph<Maxpool2x2BCHW,{self.INP_W},{self.OUT_W},{self.B},{self.C}> {self.name};"


class CppGenerator:

  def __init__(self,
               onnx_path: str,
               input_tensor: np.ndarray,
               output_tensors: Mapping[str, np.ndarray]):
    self.op_list: List[OpParser] = []
    self.nodeout_2_adfport: Mapping[str, OpParser] = {}
    self.adf_connects: List[str] = []
    
    model = onnx.load(onnx_path)
    self.nodes = model.graph.node
    
    # save input
    np.savetxt(f"../data/input.txt", input_tensor.reshape(-1, 2))
    
    # register node output and model outputs
    for i, model_input in enumerate(model.graph.input):
      self.nodeout_2_adfport[model_input.name] = f"plin[{i}].out[0]"
    self.modelout_2_adfport = {i.name: None for i in model.graph.output}

    # store I/O tensors and model parameters
    self.input_tensor: np.ndarray = input_tensor
    self.initializers: Mapping[str, np.ndarray] = {
      init.name: init for init in model.graph.initializer}
    self.output_tensors: Mapping[str, np.ndarray] = output_tensors
  
  def get_tensor(self, name: str):
    if name == "input":
      return self.input_tensor
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
        
        op = ConvOp(f"kconv{i}")
        op.save_output(toutput)
        op.register_params(tinput, tweight, tbias, toutput)
        op.register_weights(tweight, tbias)
        self.op_list.append(op)

        self.adf_connects.append(
          f"adf::connect<> ({self.nodeout_2_adfport[node.input[0]]}, {op.name}.pin[0]);")
        self.nodeout_2_adfport[onnx_out_name] = f"{op.name}.pout[0]"
        if onnx_out_name in self.modelout_2_adfport:
          self.modelout_2_adfport[onnx_out_name] = op.name
      
      elif node.op_type == "Relu":
        # handled by fusing with previous
        continue
      
      elif node.op_type == "MaxPool":
        onnx_out_name = node.output[0]
        tinput = self.get_tensor(node.input[0])
        toutput = self.get_tensor(onnx_out_name)
        op = PoolOp(f"kpool{i}")
        op.save_output(toutput)
        op.register_params(tinput, toutput)
        self.op_list.append(op)

        self.adf_connects.append(
          f"adf::connect<> ({self.nodeout_2_adfport[node.input[0]]}, {op.name}.pin[0]);")
        self.nodeout_2_adfport[onnx_out_name] = f"{op.name}.pout[0]"
        if onnx_out_name in self.modelout_2_adfport:
          self.modelout_2_adfport[onnx_out_name] = op.name

      elif node.op_type == "Gemm":
        if self.nodes[i+1].op_type != "Relu": # lookahead
          raise NotImplementedError("No relu found after Gemm, no valid implementation.")
        onnx_out_name = self.nodes[i+1].output[0]
        tinput = self.get_tensor(node.input[0])
        tweight = self.get_tensor(node.input[1])
        tbias = self.get_tensor(node.input[2])
        toutput = self.get_tensor(onnx_out_name)
        op = GemmOp(f"kgemm{i}")
        op.save_output(toutput)
        op.register_params(tinput, tweight, tbias, toutput)
        op.register_weights(tweight, tbias)
        self.op_list.append(op)

        self.adf_connects.append(
          f"for (int i = 0; i < {op.get_kernel_type()}::CHUNK_COUNT; i++)\n" + \
          f"  adf::connect<> ({self.nodeout_2_adfport[node.input[0]]}, {op.name}.pin[i]);")
        self.nodeout_2_adfport[onnx_out_name] = f"{op.name}.pout[0]"
        if onnx_out_name in self.modelout_2_adfport:
          self.modelout_2_adfport[onnx_out_name] = op.name
        
      elif node.op_type in ["Shape", "Constant", "Gather", "Unsqueeze", "Concat", "Reshape"]:
        print(f"WARNING: {node.op_type} not implemented, skipping...")
        if len(node.output[0]) != 0 and \
          np.all(self.get_tensor(node.output[0]).flatten() == toutput.flatten()):
          print(f"Found matching output {node.output[0]} and {op.name} output")
          self.nodeout_2_adfport[node.output[0]] = f"{op.name}.pout[0]"
      
      else:
        raise ValueError(f"Unexpected op_type {node.op_type}")

  def get_includes(self) -> str:
    include_list = set(i.get_include_line() for i in self.op_list)
    return "\n".join(include_list)
  
  def get_kernels(self) -> str:
    return "    " + "\n".join(i.get_kernel_line() for i in self.op_list).replace("\n", "\n    ")

  def get_args(self) -> str:
    args = [f"const std::string& {opname}_out" for opname in self.modelout_2_adfport.values()]
    args += [i.get_arg_line() for i in self.op_list]
    args += [f"const std::string& {op.name}_out = std::string()" for op in self.op_list if op.name not in self.modelout_2_adfport.values()]
    args = [i for i in args if i != ""]
    return "      " + ",\n".join(args).replace("\n", "\n      ")
  
  def get_initlist(self) -> str:
    initlists = [i.get_initlist_line() for i in self.op_list]
    initlists = [i for i in initlists if i != ""]
    return "      " + ",\n".join(initlists).replace("\n", "\n      ")
  
  def get_plouts(self) -> str:
    plouts = [
      f'adf::output_plio a = adf::output_plio::create("plout0_"+id+"_{opname}", PLIO64_ARG({opname}_out));\n' + \
      f"plout.push_back(a);\n" + \
      f"adf::connect<> ({opname}.pout[0], a.in[0]);"
      for opname in self.modelout_2_adfport.values()
    ]
    return "      " + "\n".join(plouts).replace("\n", "\n      ")
  
  def get_optional_plouts(self) -> str:
    optplouts = [
      f'SET_OPT_PLOUT({op.name}_out, adf::connect<> ({op.name}.pout[0], a.in[0]), "{op.name}");'
      for op in self.op_list if op.name not in self.modelout_2_adfport.values()]
    return "      " + "\n".join(optplouts).replace("\n", "\n      ")
  
  def get_interkernel_connects(self) -> str:
    return "      " + "\n".join(self.adf_connects).replace("\n", "\n      ")
  
  def get_weights(self) -> str:
    weights = [i.get_weight_line() for i in self.op_list]
    weights = [i for i in weights if i != ""]
    return "\n".join(weights)
  
  def get_callargs(self) -> str:
    args = ['"input.txt"']
    args += [f'"{opname}_goldenout.txt"' for opname in self.modelout_2_adfport.values()]
    args += [i.get_callarg_line() for i in self.op_list]
    args += [f'"{op.name}_goldenout.txt"' for op in self.op_list if op.name not in self.modelout_2_adfport.values()]
    args = [i for i in args if i != ""]
    return "  " + ",\n".join(args).replace("\n", "\n  ")
  
  def generate_raw(self):
    return f""" 
#include <adf.h>
{self.get_includes()}
#include "graph_utils.h"


class MyGraph : public adf::graph {{

  private:
{self.get_kernels()}

  public:
    adf::input_plio plin[1];
    std::vector<adf::output_plio> plout; // intermediate outputs optional

    MyGraph(
      const std::string& id,
      const std::string& INPUT_TXT,
{self.get_args()}
    ): 
{self.get_initlist()}
    {{ 
      // plin[0] mandatory input
      plin[0] = adf::input_plio::create("plin0_"+id+"_input", PLIO64_ARG(INPUT_TXT));
      
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
MyGraph myGraph (
  "myGraph",
{self.get_callargs()}
);


#ifdef __X86SIM__
int main(int argc, char ** argv) {{
	adfCheck(myGraph.init(), "init myGraph");
  adfCheck(myGraph.run(ITER_CNT), "run myGraph");
	adfCheck(myGraph.end(), "end myGraph");
  return 0;
}}
#endif


#ifdef __AIESIM__
int main(int argc, char ** argv) {{
	adfCheck(myGraph.init(), "init myGraph");
  get_graph_throughput_by_port(myGraph, "plout[0]", myGraph.plout[0], 1*10, sizeof(float), ITER_CNT);
	adfCheck(myGraph.end(), "end myGraph");
  return 0;
}}
#endif
"""
  
  def generate_cpp(self):
    with open("../design/aie_src/graph_gen.cpp", "w") as f:
      f.write(self.generate_raw())
