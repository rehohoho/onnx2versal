from op_parsers import dtype_to_cstr
from parser import Parser


class CppGenerator:

  def __init__(self,
               p: Parser):
    self.p = p

    gmio_allocs = []
    gmio_cpys = []
    gmio_xfers = []
    gmio_frees = []
    
    for op in p.op_list:
      for gmio_name, tensor in op.gmioname_2_tensor.items():
        ctype = dtype_to_cstr(tensor.dtype)
        size = tensor.size
        repeats = op.gmio_repeats
        
        gmio_allocs.append(
          f"{ctype}* {gmio_name}_buf = ({ctype} *) adf::GMIO::malloc({repeats}*{size}*sizeof({ctype}));"
        )
        gmio_cpys.append(
          f"for (int i = 0; i < {repeats}; i++)\n"
          f"  memcpy({gmio_name}_buf + i*{size}, {gmio_name}.data(), {size}*sizeof({ctype}));"
        )
        gmio_xfers.append(
          f"{p.graph_name}.gmio_{gmio_name}.gm2aie_nb({gmio_name}_buf, {repeats}*{size}*sizeof({ctype}));"
        )
        gmio_frees.append(
          f"adf::GMIO::free({gmio_name}_buf);"
        )
    
    self.gmio_allocs = "  " + "\n".join(gmio_allocs).replace("\n", "\n  ")
    self.gmio_cpys = "  " + "\n".join(gmio_cpys).replace("\n", "\n  ")
    self.gmio_xfers = "  " + "\n".join(gmio_xfers).replace("\n", "\n  ")
    self.gmio_frees = "  " + "\n".join(gmio_frees).replace("\n", "\n  ")
  
  def get_includes(self) -> str:
    include_list = set(i.get_include_line() for i in self.p.op_list)
    return "\n".join(include_list)
  
  def get_kernels(self) -> str:
    return "    " + "\n".join(i.get_kernel_line() for i in self.p.op_list).replace("\n", "\n    ")

  def get_input_ports(self) -> str:
    plins = [f"adf::input_plio plin[{len(self.p.modelin_2_tensor)}];"]
    for op in self.p.op_list:
      plins += [f"adf::input_gmio gmio_{gmio_name};"
                for gmio_name in op.gmioname_2_tensor]
    return "    " + "\n".join(plins).replace("\n", "\n    ")

  def get_args(self) -> str:
    args = [f"const std::string& {inp_name}" for inp_name in self.p.modelin_2_tensor]
    args += [f"const std::string& {op.name}_out" for op in self.p.modelout_2_op.values()]
    args += [op.get_arg_line() for op in self.p.op_list]
    args += [f"const std::string& {op.name}_out = std::string()" for op in self.p.op_list 
             if op not in self.p.modelout_2_op.values()]
    args = [i for i in args if i != ""]
    return "      " + ",\n".join(args).replace("\n", "\n      ")
  
  def get_initlist(self) -> str:
    initlists = [i.get_initlist_line() for i in self.p.op_list]
    initlists = [i for i in initlists if i != ""]
    return "      " + ",\n".join(initlists).replace("\n", "\n      ")
  
  def get_input_port_defs(self) -> str:
    plins = [
      f'plin[{i}] = adf::input_plio::create("plin{i}_"+id+"_{inp_name}", PLIO64_ARG({inp_name}));'
      for i, inp_name in enumerate(self.p.modelin_2_tensor)
    ]
    for op in self.p.op_list:
      for gmio_name in op.gmioname_2_tensor:
        plins.append(
          f'gmio_{gmio_name} = adf::input_gmio::create("gmio_"+id+"_{gmio_name}", 64, 1000);'
        )
    return "      " + "\n".join(plins).replace("\n", "\n      ")
  
  def get_output_port_defs(self) -> str:
    plouts = [
      f'adf::output_plio a = adf::output_plio::create("plout0_"+id+"_{op.name}", PLIO64_ARG({op.name}_out));\n' + \
      f"plout.push_back(a);\n" + \
      f"adf::connect<> ({op.name}.pout[0], a.in[0]);"
      for op in self.p.modelout_2_op.values()
    ]
    return "      " + "\n".join(plouts).replace("\n", "\n      ")
  
  def get_optional_plouts(self) -> str:
    optplouts = [
      f'SET_OPT_PLOUT({op.name}_out, adf::connect<> ({op.name}.pout[0], a.in[0]), "{op.name}");'
      for op in self.p.op_list if op not in self.p.modelout_2_op.values()]
    return "      " + "\n".join(optplouts).replace("\n", "\n      ")
  
  def get_interkernel_connects(self) -> str:
    return "      " + "\n".join(self.p.adf_connects).replace("\n", "\n      ")
  
  def get_weights(self) -> str:
    weights = [i.get_weight_line() for i in self.p.op_list]
    weights = [i for i in weights if i != ""]
    return "\n".join(weights)
  
  def get_callargs(self, is_dout: bool) -> str:
    args = [f'"{self.p.get_filename(inp_name)}"' for inp_name in self.p.modelin_2_tensor]
    args += [f'"{op.get_output_filename()}"' for op in self.p.modelout_2_op.values()]
    args += [op.get_callarg_line() for op in self.p.op_list]
    if is_dout:
      args += [f'"{op.get_output_filename()}"' for op in self.p.op_list 
               if op not in self.p.modelout_2_op.values()]
    args = [i for i in args if i != ""]
    return "  " + ",\n".join(args).replace("\n", "\n  ")
  
  def generate_cpp_graph_str(self):
    return f""" 
#include <adf.h>
{self.get_includes()}
#include "graph_utils.h"


class {self.p.graph_name.capitalize()} : public adf::graph {{

  private:
{self.get_kernels()}

  public:
{self.get_input_ports()}
    std::vector<adf::output_plio> plout; // intermediate outputs optional

    {self.p.graph_name.capitalize()}(
      const std::string& id,
{self.get_args()}
    ): 
{self.get_initlist()}
    {{ 
      // mandatory input
{self.get_input_port_defs()}

      // mandatory output
{self.get_output_port_defs()}

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
{self.p.graph_name.capitalize()} {self.p.graph_name} (
  "{self.p.graph_name}",
{self.get_callargs(is_dout=True)}
);
#else
{self.p.graph_name.capitalize()} {self.p.graph_name} (
  "{self.p.graph_name}",
{self.get_callargs(is_dout=False)}
);
#endif


#if defined(__X86SIM__) || defined(__AIESIM__)
int main(int argc, char ** argv) {{

{self.gmio_allocs}
{self.gmio_cpys}
  
  adfCheck({self.p.graph_name}.init(), "init {self.p.graph_name}");
  
{self.gmio_xfers}
  adfCheck({self.p.graph_name}.run(ITER_CNT), "run {self.p.graph_name}");
  
  adfCheck({self.p.graph_name}.end(), "end {self.p.graph_name}");
{self.gmio_frees}
  return 0;
}}
#endif
"""
  
  def generate_cpp_graph(self):
    with open(f"../design/aie_src/graph_{self.p.graph_name}.cpp", "w") as f:
      f.write(self.generate_cpp_graph_str())
