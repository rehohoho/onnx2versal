from typing import List, Mapping
import os

import numpy as np
import onnx
from onnx import numpy_helper
# from google.protobuf.json_format import MessageToJson, MessageToDict

from op_parsers import dtype_to_cstr, save_tensor, pad_lastdim, OpParser, \
  ArgmaxOp, ConvOp, DequantizeLinearOp, GemmOp, PoolOp, QGemm, QLinearConvOp, QuantizeLinearOp, QLinearSoftmaxOp, SoftmaxOp


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
    for input_name, input_tensor in self.modelin_2_tensor.items():
      save_tensor(f"{self.data_path}/{input_name}.txt", input_tensor)

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
    
  def register_port(self, onnx_in_name: str, onnx_out_name: str, op: OpParser):
    self.adf_connects.append(op.get_connect_line(self.nodeout_2_adfport[onnx_in_name]))
    self.nodeout_2_adfport[onnx_out_name] = f"{op.name}.pout[0]"
    if onnx_out_name in self.modelout_2_op:
      self.modelout_2_op[onnx_out_name] = op

  # assumes nodes are in topological sorted order
  def parse(self):
    i = 0
    while i < len(self.nodes):
      node = self.nodes[i]
      
      if node.op_type == "Conv":
        if self.nodes[i+1].op_type != "Relu": # lookahead
          raise NotImplementedError("No relu found after Conv, no valid implementation.")
        print(f"WARNING: fusing Conv+Relu")

        onnx_out_name = self.nodes[i+1].output[0]
        op = ConvOp(f"k{i}conv")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, onnx_out_name)])
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port(node.input[0], onnx_out_name, op)
        i += 1
      
      elif node.op_type == "MaxPool":
        op = PoolOp(f"k{i}pool")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)])
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port(node.input[0], node.output[0], op)

      elif node.op_type == "Gemm":
        is_relu = self.nodes[i+1].op_type == "Relu" # lookahead
        
        if is_relu:
          onnx_out_name = self.nodes[i+1].output[0]
        else:
          onnx_out_name = node.output[0]

        op = GemmOp(f"k{i}gemm", is_relu=is_relu)
        op.register_params([self.get_tensor(tname) for tname in (*node.input, onnx_out_name)], 
                           node.attribute)
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port(node.input[0], onnx_out_name, op)
        
        if is_relu:
          i += 1
      
      elif node.op_type == "QuantizeLinear":
        op = QuantizeLinearOp(f"k{i}quantizelinear")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)])
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port(node.input[0], node.output[0], op)

      elif node.op_type == "QLinearConv":
        op = QLinearConvOp(f"k{i}qlinearconv")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)])
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port(node.input[0], node.output[0], op)
        
      elif node.op_type == "QGemm":
        op = QGemm(f"k{i}qgemm")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)],
                           node.attribute)
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port(node.input[0], node.output[0], op)
      
      elif node.op_type == "DequantizeLinear":
        op = DequantizeLinearOp(f"k{i}dequantizeLinear")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)])
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port(node.input[0], node.output[0], op)

      elif node.op_type in ["Shape", "Constant", "Gather", "Unsqueeze", "Concat", "Reshape"]:
        if len(node.output[0]) != 0 and np.all(self.get_tensor(node.output[0]).flatten() == self.op_list[-1].tout.flatten()):
          print(f"Found matching output {node.output[0]} and {op.name} output")
          self.nodeout_2_adfport[node.output[0]] = f"{op.name}.pout[0]"
          self.op_list[-1].disable_output_pad()
        else:
          print(f"WARNING: {node.op_type} not implemented, skipping...")
      
      elif node.op_type == "MatMul":
        if self.nodes[i+1].op_type != "Add": # lookahead
          raise NotImplementedError("No Add found after MatMul, no valid implementation.")
        
        is_relu = self.nodes[i+2].op_type == "Relu"
        
        if is_relu:
          print(f"WARNING: fusing MatMul+Add+Relu")
          onnx_out_name = self.nodes[i+2].output[0]
        else:
          onnx_out_name = self.nodes[i+1].output[0]
        
        bias_name = self.nodes[i+1].input[1]
        op = GemmOp(f"k{i}gemm", is_relu=is_relu)
        op.register_params([self.get_tensor(tname) for tname in (*node.input, bias_name, onnx_out_name)],
                           node.attribute)
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port(node.input[0], onnx_out_name, op)

        i += 1 # add
        if is_relu: i += 1 # relu
      
      elif node.op_type == "Softmax":
        op = SoftmaxOp(f"k{i}softmax")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)])
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port(node.input[0], node.output[0], op)
      
      elif node.op_type == "QLinearSoftmax":
        op = QLinearSoftmaxOp(f"k{i}qlinearsoftmax")
        op.register_params([self.get_tensor(tname) for tname in (*node.input, *node.output)], node.attribute)
        op.save_txt(self.data_path)
        self.op_list.append(op)
        self.register_port(node.input[0], node.output[0], op)
      
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
    args = [f'"{inpname}.txt"' for inpname in self.modelin_2_tensor]
    args += [f'"{list(op.filename_2_tensors.keys())[-1]}"' for op in self.modelout_2_op.values()]
    args += [i.get_callarg_line() for i in self.op_list]
    if is_output_inter:
      args += [f'"{list(op.filename_2_tensors.keys())[-1]}"' for op in self.op_list 
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
    file_prefix = ""
    if not is_output_inter:
      file_prefix = "host_"
    masters = [
      f'("plin{i}_{self.graph_name}_{inpname}", f"{{args.input_dir}}/{file_prefix}{inpname}.txt", 64, ' + \
      f'"{str(self.modelin_2_tensor[inpname].dtype)}")'
      for i, inpname in enumerate(self.modelin_2_tensor)
    ]
    return "    " + ",\n".join(masters).replace("\n", "\n    ")
  
  def get_xtg_slaves(self, is_output_inter: bool) -> str:
    slaves = []
    for out_name, op in self.modelout_2_op.items():
      size = op.out_size
      if not is_output_inter:
        size *= self.data_count
        out_name = f"host_{out_name}"
      slaves += [
        f'("plout{i}_{self.graph_name}_{op.name}", f"{{args.output_dir}}/{out_name}.txt", ' + \
        f'64, "{str(op.dtype)}", {size})'
        for i, op in enumerate(self.modelout_2_op.values())
      ]

    if is_output_inter:
      i = len(self.modelout_2_op)
      for op in self.op_list:
        if op in self.modelout_2_op.values(): continue
        slaves.append(
          f'("plout{i}_{self.graph_name}_{op.name}", f"{{args.output_dir}}/{list(op.filename_2_tensors.keys())[-1]}", 64, "{str(op.dtype)}", {op.out_size})')
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
    mm2s_names = {}
    for i, tensor in enumerate(self.modelin_2_tensor.values()):
      dtype = tensor.dtype
      if dtype not in mm2s_names:
        mm2s_names[dtype] = []
      mm2s_names[dtype].append(f"{dtype}_mm2s_{i}")
    
    header = ""
    for dtype, typed_mm2s_names in mm2s_names.items():
      header += f"nk={dtype}_mm2s:{len(typed_mm2s_names)}:{','.join(typed_mm2s_names)}"
      header += "\n"
    
    return header
  
  def get_cfg_output_kernels(self, is_output_inter: bool) -> str:
    s2mm_names = {}
    ops = list(self.modelout_2_op.values())
    if is_output_inter:
      ops += [op for op in self.op_list if op not in self.modelout_2_op.values()]
    for i, op in enumerate(ops):
      dtype = str(op.dtype)
      if dtype not in s2mm_names: 
        s2mm_names[dtype] = []
      s2mm_names[dtype].append(f"{dtype}_s2mm_{i}")

    header = ""
    for dtype, typed_s2mm_names in s2mm_names.items():
      header += f"nk={dtype}_s2mm:{len(typed_s2mm_names)}:{','.join(typed_s2mm_names)}"
      header += "\n"
    
    return header
  
  def get_cfg_input_scs(self) -> str:
    in_scs = [
      f"stream_connect={tensor.dtype}_mm2s_{i}.s:ai_engine_0.plin{i}_{self.graph_name}_{inpname}"
      for i, (inpname, tensor) in enumerate(self.modelin_2_tensor.items())
    ]
    return "\n".join(in_scs)

  def get_cfg_output_scs(self, is_output_inter: bool) -> str:
    ops = list(self.modelout_2_op.values())
    if is_output_inter:
      ops += [op for op in self.op_list if op not in self.modelout_2_op.values()]
    out_scs = [
      f"stream_connect=ai_engine_0.plout{i}_{self.graph_name}_{op.name}:{op.dtype}_s2mm_{i}.s"
      for i, op in enumerate(ops)
    ]
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
      f'#define OUTPUT{i}_FILENAME "{out_name}.txt"'
      for i, out_name in enumerate(self.modelout_2_op.keys())
    ]
    outfiles_host = [i.replace('FILENAME "', 'FILENAME "host_') for i in outfiles]
    outfiles = ["#ifdef __OUTPUT_INTER__"] + outfiles + ["#else"] + outfiles_host + ["#endif"]
    
    n_outs = len(self.modelout_2_op)
    outfiles += [
      f'#define INTER{n_outs+i}_FILENAME "{list(op.filename_2_tensors.keys())[-1]}"'
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
write_arr_to_file<{dtype_to_cstr(op.dtype)}>(out_dir+OUTPUT{i}_FILENAME, out{i}_bomapped, iter_cnt*{op.out_size});
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
      optout_writes.append(f"""write_arr_to_file<{dtype_to_cstr(op.dtype)}>(out_dir+INTER{i}_FILENAME, inter{i}_bomapped, iter_cnt*{op.out_size});
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


template <typename TT>
void write_arr_to_file(
   const std::string& filename,
   const TT* bomapped,
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
