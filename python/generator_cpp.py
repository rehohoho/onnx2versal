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

    gmio_buf_allocs = []
    gmio_buf_xfers = []
    
    includes = []
    kernels = []
    plins = []
    gmios = []
    args = []
    inits = []
    plindefs = []
    gmiodefs = []
    optional_plouts = []
    weights = []

    plins += [f"adf::input_plio plin[{len(self.p.modelin_2_tensor)}];"]
    optional_args = []

    for id, input_name, tensor in self.p.get_input_id_name_tensor():
      args.append(f"const std::string& {input_name}")
      plindefs.append(f'plin[{id}] = adf::input_plio::create("plin{id}_"+id+"_{input_name}", PLIO64_ARG({input_name}));')
    
    for id, op in self.p.get_output_id_op():
      args.append(f"const std::string& {op.name}_out")
    
    for i, (onnx_name, op) in enumerate(p.onnxname_2_op.items()):
      if i == 0: # skip input node
        continue
      
      includes.append(op.get_include_line())
      kernels.append(op.get_kernel_line())
      args.append(op.get_arg_line())
      inits.append(op.get_initlist_line())
      weights.append(op.get_weight_line())

      if op not in self.p.modelout_2_op.values():
        optional_args.append(f"const std::string& {op.name}_out = std::string()")
        optional_plouts.append(
          f'SET_OPT_PLOUT({op.name}_out, adf::connect<> ({op.get_adf_port_name()}, a.in[0]), "{op.name}");'  
        )

      for gmio_name, tensor in op.gmioname_2_tensor.items():
        ctype = dtype_to_cstr(tensor.dtype)
        size = tensor.size
        repeats = op.gmio_repeats

        gmios.append(
          f"adf::input_gmio  gmio_{gmio_name};"
        )
        gmiodefs.append(
          f'gmio_{gmio_name} = adf::input_gmio::create("gmio_"+id+"_{gmio_name}", 64, 500);'
        )
        gmio_allocs.append(
          f"{ctype}* {gmio_name}_buf = ({ctype} *) adf::GMIO::malloc({repeats}*{size}*ITER_CNT*sizeof({ctype}));"
        )
        gmio_cpys.append(
          f"for (int i = 0; i < {repeats}*ITER_CNT; i++)\n"
          f"  memcpy({gmio_name}_buf + i*{size}, {gmio_name}.data(), {size}*sizeof({ctype}));"
        )
        gmio_xfers.append(
          f"{p.graph_name}.gmio_{gmio_name}.gm2aie_nb({gmio_name}_buf, {repeats}*{size}*ITER_CNT*sizeof({ctype}));"
        )
        gmio_frees.append(
          f"adf::GMIO::free({gmio_name}_buf);"
        )
      
      for gmio_name, bufname in op.gmioin_2_bufname.items():
        gmio_buf_dtype, gmio_buf_size = p.gmiobuf_2_size[bufname]
        gmios.append(f"adf::input_gmio  gmio_{gmio_name};")
        gmiodefs.append(f'gmio_{gmio_name} = adf::input_gmio::create("gmio_"+id+"_{gmio_name}", 64, 500);')
        gmio_buf_xfers.append(
          f"{p.graph_name}.gmio_{gmio_name}.gm2aie_nb({bufname}, 1*{gmio_buf_size}*ITER_CNT*sizeof({gmio_buf_dtype}));"
        )
      
      for gmio_name, bufname in op.gmioout_2_bufname.items():
        gmio_buf_dtype, gmio_buf_size = p.gmiobuf_2_size[bufname]
        gmios.append(f"adf::output_gmio gmio_{gmio_name};")
        gmiodefs.append(f'gmio_{gmio_name} = adf::output_gmio::create("gmio_"+id+"_{gmio_name}", 64, 500);')
        gmio_buf_xfers.append(
          f"{p.graph_name}.gmio_{gmio_name}.aie2gm({bufname}, 1*{gmio_buf_size}*ITER_CNT*sizeof({gmio_buf_dtype}));"
        )
    
    for gmio_buf_name, (gmio_buf_dtype, gmio_buf_size) in p.gmiobuf_2_size.items():
      gmio_buf_allocs.append(
        f"{gmio_buf_dtype}* {gmio_buf_name} = ({gmio_buf_dtype} *) adf::GMIO::malloc(1*{gmio_buf_size}*ITER_CNT*sizeof({gmio_buf_dtype}));"
      )
    
    args += optional_args
    
    self.includes = "\n".join(set(includes))
    self.kernels = "    " + "\n".join(kernels).replace("\n", "\n    ")
    self.plins = "    " + "\n".join(plins).replace("\n", "\n    ")
    self.gmios = "    " + "\n".join(gmios).replace("\n", "\n    ")
    self.args = "      " + ",\n".join(i for i in args if i != "").replace("\n", "\n      ")
    self.initlists = "      " + ",\n".join(i for i in inits if i != "").replace("\n", "\n      ")
    self.plindefs = "      " + "\n".join(plindefs).replace("\n", "\n      ")
    self.gmiodefs = "      " + "\n".join(gmiodefs).replace("\n", "\n      ")
    self.optional_plouts = "      " + "\n".join(optional_plouts).replace("\n", "\n      ")
    self.weights = "\n".join(i for i in weights if i != "")

    self.gmio_allocs = "  " + "\n".join(gmio_allocs).replace("\n", "\n  ")
    self.gmio_buf_allocs = "  " + "\n".join(gmio_buf_allocs).replace("\n", "\n  ")
    self.gmio_cpys = "  " + "\n".join(gmio_cpys).replace("\n", "\n  ")
    self.gmio_xfers = "  " + "\n".join(gmio_xfers).replace("\n", "\n  ")
    self.gmio_buf_xfers = "  " + "\n".join(gmio_buf_xfers).replace("\n", "\n  ")
    self.gmio_frees = "  " + "\n".join(gmio_frees).replace("\n", "\n  ")
  
  def get_output_plindefs(self) -> str:
    plouts = [
      f'adf::output_plio a = adf::output_plio::create("plout0_"+id+"_{op.name}", PLIO64_ARG({op.name}_out));\n' + \
      f"plout.push_back(a);\n" + \
      f"adf::connect<> ({op.name}.pout[0], a.in[0]);"
      for op in self.p.modelout_2_op.values()
    ]
    return "      " + "\n".join(plouts).replace("\n", "\n      ")
  
  def get_interkernel_connects(self) -> str:
    return "      " + "\n".join(self.p.adf_connects).replace("\n", "\n      ")
  
  def get_callargs(self, is_dout: bool) -> str:
    args = [f'"{fn}"' for fn in self.p.get_input_filename(is_dout)]
    args += [f'"{fn}"' for fn in self.p.get_output_filename(is_dout)]
    args += [op.get_callarg_line() for op in self.p.onnxname_2_op.values()]
    
    if is_dout:
      optargs = []
      for i, op in enumerate(self.p.onnxname_2_op.values()):
        if op not in self.p.modelout_2_op.values() and i != 0: # skip input node
          fn = f'"{op.get_output_filename()}"'
          optargs.append(fn)
      args += optargs
    args = [i for i in args if i != ""]
    return "  " + ",\n".join(args).replace("\n", "\n  ")
  
  def generate_cpp_graph_str(self):
    return f""" 
#include <adf.h>
{self.includes}
#include "graph_utils.h"


class {self.p.graph_name.capitalize()} : public adf::graph {{

  private:
{self.kernels}

  public:
{self.plins}
{self.gmios}
    std::vector<adf::output_plio> plout; // intermediate outputs optional

    {self.p.graph_name.capitalize()}(
      const std::string& id,
{self.args}
    ): 
{self.initlists}
    {{ 
      // mandatory input
{self.plindefs}
{self.gmiodefs}

      // mandatory output
{self.get_output_plindefs()}

#define SET_OPT_PLOUT(TXT_PATH, STMT, PLOUT_NAME) \\
      if (!TXT_PATH.empty()) {{ \\
        std::string plout_name = "plout"+std::to_string(plout.size())+"_"+id+"_"+PLOUT_NAME; \\
        adf::output_plio a = adf::output_plio::create(plout_name, PLIO64_ARG(TXT_PATH)); \\
        STMT; plout.push_back(a);}} 

      // optional output
{self.optional_plouts}

      // interkernel
{self.get_interkernel_connects()}
      
    }}
}};

{self.weights}

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
{self.gmio_buf_allocs}
{self.gmio_cpys}
  
  adfCheck({self.p.graph_name}.init(), "init {self.p.graph_name}");
  
{self.gmio_xfers}
  adfCheck({self.p.graph_name}.run(ITER_CNT), "run {self.p.graph_name}");
{self.gmio_buf_xfers}
  
  adfCheck({self.p.graph_name}.end(), "end {self.p.graph_name}");
{self.gmio_frees}
  return 0;
}}
#endif
"""
  
  def generate_cpp_graph(self):
    with open(f"../design/aie_src/graph_{self.p.graph_name}.cpp", "w") as f:
      f.write(self.generate_cpp_graph_str())
