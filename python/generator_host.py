from op_parsers import dtype_to_cstr, InputOp
from parser import Parser


class HostGenerator:

  def __init__(self,
               parser: Parser):
    self.p = parser

  def get_host_datafiles(self) -> str:
    outfiles = ["#ifdef __OUTPUT_INTER__"]
    outfiles += [
      f'#define INPUT{id}_FILENAME "{self.p.get_filename(input_name)}"'
      for id, input_name, tensor in self.p.get_input_id_name_tensor()
    ]
    outfiles += [f'#define OUTPUT{i}_FILENAME "{fn}"' for i, fn in enumerate(self.p.get_output_filename(True))]
    outfiles += ["#else"]
    outfiles += [f'#define INPUT{i}_FILENAME "{fn}"' for i, fn in enumerate(self.p.get_input_filename(False))]
    outfiles += [f'#define OUTPUT{i}_FILENAME "{fn}"' for i, fn in enumerate(self.p.get_output_filename(False))]
    outfiles += ["#endif"]
    
    n_outs = len(self.p.modelout_2_op)
    outfiles += [
      f'#define INTER{n_outs+id}_FILENAME "{op.get_output_filename()}"'
      for id, op in self.p.get_output_id_op(include_output=False, include_optional_output=True)
    ]
    return "\n".join(outfiles)
  
  def get_host_input_inits(self) -> str:
    inp_inits = []
    inp_initsyncs = ["", "#ifdef __IS_SW_EMU__"]
    
    for id, input_name, tensor in self.p.get_input_id_name_tensor():
      size = tensor.size
      dtype = tensor.dtype
      ctype = dtype_to_cstr(dtype)
      inp_inits.append(f"""xrtBufferHandle in{id}_bohdl = xrtBOAlloc(dhdl, iter_cnt*{size}*sizeof({ctype}), 0, 0);
auto in{id}_bomapped = reinterpret_cast<{ctype}*>(xrtBOMap(in{id}_bohdl));
printf("Input{id} memory virtual addr 0x%p\\n", in{id}_bomapped);
read_arr_from_file(data_dir+INPUT{id}_FILENAME, in{id}_bomapped, iter_cnt*{size});

xrtKernelHandle in{id}_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "{dtype}_mm2s:{{{dtype}_mm2s_{id}}}");
xrtRunHandle in{id}_rhdl = xrtRunOpen(in{id}_khdl); 
xrtRunSetArg(in{id}_rhdl, 0, in{id}_bohdl);
xrtRunSetArg(in{id}_rhdl, 2, iter_cnt*{size});
xrtRunStart(in{id}_rhdl);""")
      inp_initsyncs.append(
        f"xrtBOSync(in{id}_bohdl, XCL_BO_SYNC_BO_TO_DEVICE, iter_cnt*{size}*sizeof({ctype}), 0);"
      )
    
    inp_initsyncs.append("#endif")
    return "   " + "\n".join(inp_inits+inp_initsyncs).replace("\n", "\n   ")
  
  def get_host_input_closes(self) -> str:
    inp_closes = [f"""auto in{id}_state = xrtRunWait(in{id}_rhdl);    
printf("mm2s completed with status (%d)\\n", in{id}_state);
xrtRunClose(in{id}_rhdl);
xrtKernelClose(in{id}_khdl);
xrtBOFree(in{id}_bohdl);"""
      for id, input_name, tensor in self.p.get_input_id_name_tensor()
    ]
    return "   " + "\n".join(inp_closes).replace("\n", "\n   ")
  
  def get_host_output_inits(self) -> str:
    out_inits = []
    for id, op in self.p.get_output_id_op():
      ctype = dtype_to_cstr(op.dtype)
      out_inits.append(f"""xrtBufferHandle out{id}_bohdl = xrtBOAlloc(dhdl, iter_cnt*{op.out_size}*sizeof({ctype}), 0, 0);
auto out{id}_bomapped = reinterpret_cast<{ctype}*>(xrtBOMap(out{id}_bohdl));
printf("Output{id} memory virtual addr 0x%p\\n", out{id}_bomapped);

xrtKernelHandle out{id}_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "{op.dtype}_s2mm:{{{op.dtype}_s2mm_{id}}}");
xrtRunHandle out{id}_rhdl = xrtRunOpen(out{id}_khdl); 
xrtRunSetArg(out{id}_rhdl, 0, out{id}_bohdl);
xrtRunSetArg(out{id}_rhdl, 2, iter_cnt*{op.out_size});
xrtRunStart(out{id}_rhdl);""")
    return "   " + "\n".join(out_inits).replace("\n", "\n   ")

  def get_host_output_closes(self) -> str:
    out_closes = [
    f"""auto out{id}_state = xrtRunWait(out{id}_rhdl);
printf("s2mm completed with status (%d)\\n", out{id}_state);
xrtRunClose(out{id}_rhdl);
xrtKernelClose(out{id}_khdl);
#ifdef __IS_SW_EMU__
xrtBOSync(out{id}_bohdl, XCL_BO_SYNC_BO_FROM_DEVICE, iter_cnt*{op.out_size}*sizeof({dtype_to_cstr(op.dtype)}), 0);
#endif
write_arr_to_file(out_dir+OUTPUT{id}_FILENAME, out{id}_bomapped, iter_cnt*{op.out_size});
xrtBOFree(out{id}_bohdl);"""
      for id, op in self.p.get_output_id_op()
    ]
    return "   " + "\n".join(out_closes).replace("\n", "\n   ")

  def get_host_optout_inits(self) -> str:
    optout_inits = []
    for id, op in self.p.get_output_id_op(include_output=False, include_optional_output=True):
      ctype = dtype_to_cstr(op.dtype)
      optout_inits.append(f"""
xrtBufferHandle inter{id}_bohdl = xrtBOAlloc(dhdl, iter_cnt*{op.out_size}*sizeof({ctype}), 0, 0);
auto inter{id}_bomapped = reinterpret_cast<{ctype}*>(xrtBOMap(inter{id}_bohdl));
printf("Inter{id} memory virtual addr 0x%p\\n", inter{id}_bomapped);
xrtKernelHandle inter{id}_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "{op.dtype}_s2mm:{{{op.dtype}_s2mm_{id}}}");
xrtRunHandle inter{id}_rhdl = xrtRunOpen(inter{id}_khdl);
xrtRunSetArg(inter{id}_rhdl, 0, inter{id}_bohdl);
xrtRunSetArg(inter{id}_rhdl, 2, iter_cnt*{op.out_size});
xrtRunStart(inter{id}_rhdl);""")
    return "   " + "\n".join(optout_inits).replace("\n", "\n   ")
  
  def get_host_optout_closes(self) -> str:
    optout_closes = []
    optout_syncs = ["#ifdef __IS_SW_EMU__"]
    optout_writes = []
    for id, op in self.p.get_output_id_op(include_output=False, include_optional_output=True):
      optout_closes.append(f"""auto inter{id}_state = xrtRunWait(inter{id}_rhdl);
printf("inter{id} completed with status (%d)\\n", inter{id}_state);
xrtRunClose(inter{id}_rhdl);
xrtKernelClose(inter{id}_khdl);""")
      optout_syncs.append(
        f"xrtBOSync(inter{id}_bohdl, XCL_BO_SYNC_BO_FROM_DEVICE, iter_cnt*{op.out_size}*sizeof({dtype_to_cstr(op.dtype)}), 0);")
      optout_writes.append(f"""write_arr_to_file(out_dir+INTER{id}_FILENAME, inter{id}_bomapped, iter_cnt*{op.out_size});
xrtBOFree(inter{id}_bohdl);""")
    optout_syncs.append("#endif")
    return "   " + "\n".join(optout_closes + optout_syncs + optout_writes).replace("\n", "\n   ")

  def generate_host_cpp_str(self) -> str:
    return f"""
#include <fstream>
#include <type_traits>

#include "graph_{self.p.graph_name}.cpp"

#include "experimental/xrt_aie.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_bo.h"
#include "experimental/xrt_error.h"

#include "adf/adf_api/XRTConfig.h"


#define PLIO_BYTEWIDTH  8
{self.get_host_datafiles()}


template <typename TT>
void read_arr_from_file(
   const std::string& filename,
   TT* bomapped,
   const size_t bosize
) {{
   std::ifstream inp_file;
   inp_file.open(filename, std::ifstream::in);
   if (!inp_file) printf("Unable to open %s.\\n", filename.c_str());
   
   int v_per_line = PLIO_BYTEWIDTH / sizeof(TT);
   TT d;
   for (int j = 0; j < bosize; j+=v_per_line) {{
     for (int k = 0; k < v_per_line; k++) {{
         inp_file >> d;
         bomapped[j+k] = d;
     }}
   }}
}}


template <typename TT>
void write_arr_to_file(
   const std::string& filename,
   TT* bomapped,
   const size_t bosize
) {{
   std::ofstream file;
   file.open(filename, std::ofstream::out);
   if (!file) printf("Unable to open %s\\n", filename.c_str());

   typedef typename std::conditional<(std::is_same<TT, float>::value), 
                                     float, int>::type fout_dtype;
   int v_per_line = PLIO_BYTEWIDTH / sizeof(TT);

   for (int j = 0; j < bosize; j+=v_per_line) {{
      for (int k = 0; k < v_per_line; k++) {{
         file << (fout_dtype) bomapped[j+k] << " ";
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
      adfCheck({self.p.graph_name}.init(), "init {self.p.graph_name}");
#ifdef __IS_SW_EMU__
      adfCheck({self.p.graph_name}.run(iter_cnt), "run {self.p.graph_name}");
      adfCheck({self.p.graph_name}.wait(), "wait {self.p.graph_name}");
#else
      get_graph_throughput_by_port({self.p.graph_name}, "plout[0]", {self.p.graph_name}.plout[0], 1*iter_cnt, sizeof(float_t), iter_cnt);
#endif
      adfCheck({self.p.graph_name}.end(), "end {self.p.graph_name}");
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
    with open(f"../design/host_app_src/{self.p.graph_name}_aie_app.cpp", "w") as f:
      f.write(self.generate_host_cpp_str())
