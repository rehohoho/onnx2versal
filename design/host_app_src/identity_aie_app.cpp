#include <fstream>

#include "graph_identity.cpp"

#include "experimental/xrt_aie.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_bo.h"
#include "experimental/xrt_error.h"

#include "adf/adf_api/XRTConfig.h"


#define INPUT_FILENAME  "concat_fpin.txt"
#define OUTPUT_FILENAME "concat_fpin.txt"
#define BUFF_SIZE      128
#define PLIO_BYTEWIDTH 8


template <typename TT>
void read_arr_from_file(
   const std::string& filename,
   TT* bomapped,
   const size_t bosize
) {
   std::ifstream inp_file;
   inp_file.open(filename, std::ifstream::in);
   if (!inp_file) printf("Unable to open %s.\n", filename.c_str());
   
   int v_per_line = PLIO_BYTEWIDTH / sizeof(TT);
   TT d;
   for (int j = 0; j < bosize; j+=v_per_line) {
     for (int k = 0; k < v_per_line; k++) {
         inp_file >> d;
         bomapped[j+k] = d;
     }
   }
}

template <typename TT>
void write_arr_to_file(
   const std::string& filename,
   TT* bomapped,
   const size_t bosize
) {
   std::ofstream file;
   file.open(filename, std::ofstream::out);
   if (!file) printf("Unable to open %s\n", filename.c_str());

   typedef typename std::conditional<(std::is_same<TT, float>::value), 
                                     float, int>::type fout_dtype;
   int v_per_line = PLIO_BYTEWIDTH / sizeof(TT);

   for (int j = 0; j < bosize; j+=v_per_line) {
      for (int k = 0; k < v_per_line; k++) {
         file << (fout_dtype) bomapped[j+k] << " ";
      }
      file << std::endl;
   }
}

static std::vector<char> load_xclbin(
   xrtDeviceHandle device, 
   const std::string &fnm
) {
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
}

int main(int argc, char ** argv) {
   // Parse args
   if(argc != 5) {
      std::cout << "Usage: " << argv[0] <<" <xclbin>" << " <iter_cnt>" << " <data_dir>" << " <out_dir>" << std::endl;
      return EXIT_FAILURE;
   }
   const char* xclbin_path = argv[1];
   const int iter_cnt = atoi(argv[2]);
   std::string data_dir = argv[3];
   data_dir.append("/");
   std::string out_dir = argv[4];
   out_dir.append("/");
   printf("\nConfig:\nxclbin: %s\niter_cnt: %d\ndata_dir: %s\nout_dir: %s\n\n", 
      xclbin_path, iter_cnt, data_dir.c_str(), out_dir.c_str());


   // Open device, load xclbin
   auto deviceIdx = xrt::device(0);
   auto dhdl = xrtDeviceOpen(0);
   auto xclbin = load_xclbin(dhdl, xclbin_path);
   auto top = reinterpret_cast<const axlf*>(xclbin.data());
   adf::registerXRT(dhdl, top->m_header.uuid);


   // Allocate BOs (buffer objects) of requested size with appropriate flags
   // Memory map BOs into user's address space (DDR Memory)
   size_t input_size_in_bytes = BUFF_SIZE * sizeof(float);
   xrtBufferHandle in_bohdl = xrtBOAlloc(dhdl, input_size_in_bytes, 0, 0);
   auto in_bomapped = reinterpret_cast<float*>(xrtBOMap(in_bohdl));
   printf("Input memory virtual addr 0x%p\n", in_bomapped);

   size_t output_size_in_bytes = BUFF_SIZE * sizeof(float);
   xrtBufferHandle out_bohdl = xrtBOAlloc(dhdl, output_size_in_bytes, 0, 0);
   auto out_bomapped = reinterpret_cast<float*>(xrtBOMap(out_bohdl));
   printf("Output memory virtual addr 0x%p\n", out_bomapped);

   
   // Read in data from file
   std::ifstream inp_file;
   read_arr_from_file(data_dir+INPUT_FILENAME, in_bomapped, iter_cnt*BUFF_SIZE);

   #ifdef __IS_SW_EMU__
      xrtBOSync(in_bohdl, XCL_BO_SYNC_BO_TO_DEVICE, input_size_in_bytes, 0);
      printf("xrtBOSync done.\n");
   #endif


   // Create kernel handle, runtime handle, set args, start kernels
   xrtKernelHandle float32_s2mm_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "float32_s2mm:{float32_s2mm_0}");
   xrtRunHandle float32_s2mm_rhdl = xrtRunOpen(float32_s2mm_khdl); 
   xrtRunSetArg(float32_s2mm_rhdl, 0, out_bohdl);
   xrtRunSetArg(float32_s2mm_rhdl, 2, BUFF_SIZE);

   xrtKernelHandle float32_mm2s_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "float32_mm2s:{float32_mm2s_0}");
   xrtRunHandle float32_mm2s_rhdl = xrtRunOpen(float32_mm2s_khdl); 
   xrtRunSetArg(float32_mm2s_rhdl, 0, in_bohdl);
   xrtRunSetArg(float32_mm2s_rhdl, 2, BUFF_SIZE);
   

   // Graph execution for AIE
   // try {
   //    adfCheck(fpscalar.init(), "init fpscalar");
   //    adfCheck(fpscalar.run(ITER_CNT), "run fpscalar");
   //    adfCheck(fpscalar.end(), "end fpscalar");
   //    // get_graph_throughput_by_port(fpscalar, "plout[0]", fpscalar.plout[0], 1*8, sizeof(float32_t), ITER_CNT);
   // }
   // catch (const std::system_error& ex) {
   //    xrt::error error(deviceIdx, XRT_ERROR_CLASS_AIE);
   //    auto errCode = error.get_error_code();
   //    auto timestamp = error.get_timestamp();
   //    auto err_str = error.to_string();
   //    std::cout << timestamp << " error code:" << errCode << " Error:" << err_str << std::endl;
   // }
   xrtGraphHandle gr;
   
   gr = xrtGraphOpen(dhdl, top->m_header.uuid, "fpscalar");
   if (!gr) 
      throw std::runtime_error("Unable to open GEMM g graph handle");
   
   int ret = xrtGraphReset(gr);
   ret = xrtGraphRun(gr, iter_cnt);
   if (ret)
      throw std::runtime_error("Unable to run GEMM g graph");
   
   
   xrtRunStart(float32_s2mm_rhdl);
   xrtRunStart(float32_mm2s_rhdl);

   printf("Waiting for dma hls to complete...\n");
   auto float32_s2mm_state = xrtRunWait(float32_s2mm_rhdl);
   auto float32_mm2s_state = xrtRunWait(float32_mm2s_rhdl);
   
   // Wait for Kernel execution to end, close runtime and kernel handlers   
   printf("float32_s2mm completed with status (%d)\n", float32_s2mm_state);
   xrtRunClose(float32_s2mm_rhdl); // xrtRunOpen
   xrtKernelClose(float32_s2mm_khdl); // xrtPLKernelOpen
   
   printf("float32_mm2s completed with status (%d)\n", float32_mm2s_state);
   xrtRunClose(float32_mm2s_rhdl);
   xrtKernelClose(float32_mm2s_khdl);
   
   printf("Closed runtime handlers and kernel handlers...\n");

   #ifdef __IS_SW_EMU__
      xrtBOSync(out_bohdl, XCL_BO_SYNC_BO_FROM_DEVICE, output_size_in_bytes, 0);
   #endif
   
   // Write and check outputs
   write_arr_to_file(out_dir+OUTPUT_FILENAME, out_bomapped, iter_cnt*BUFF_SIZE);
   
   //Release allocated resources
   xrtBOFree(in_bohdl);
   xrtBOFree(out_bohdl);
   printf("Released I/O buffer objects.\n");
   
   xrtGraphClose(gr);
   
   xrtDeviceClose(dhdl);
   
   return 0;
   
}
