#include <fstream>

#include "graph_identity.cpp"

#include "experimental/xrt_aie.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_bo.h"
#include "experimental/xrt_error.h"

#include "adf/adf_api/XRTConfig.h"


#define INPUT_SIZE   8 * ITER_CNT
#define OUTPUT_SIZE  8 * ITER_CNT


void fill_input_pl(float input_pl[]) {
   int idx = 0;
   for (int i = 0; i < INPUT_SIZE; i++) {
      input_pl[idx] = i; idx++;
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
   if((argc < 2) || (argc > 3)) {
      std::cout << "Usage: " << argv[0] <<" <xclbin>" << "iteration count(optional)" << std::endl;
      return EXIT_FAILURE;
   }
   const char* xclbinFilename = argv[1];
   int16_t iterCnt = 0;
   if(argc == 3) {
      std::string iter = argv[2];
      iterCnt = stoi(iter);
   } else {
      iterCnt = NET_INSTS;
   }
   printf("Iteration : %d...\n", iterCnt);


   // Open device, load xclbin
   auto deviceIdx = xrt::device(0);
   auto dhdl = xrtDeviceOpen(0);
   auto xclbin = load_xclbin(dhdl, xclbinFilename);
   auto top = reinterpret_cast<const axlf*>(xclbin.data());
   adf::registerXRT(dhdl, top->m_header.uuid);


#ifndef EXTERNAL_IO
   // Generate inputs
   float input_pl[INPUT_SIZE];
   fill_input_pl(input_pl);
   size_t input_size_in_bytes = INPUT_SIZE * sizeof(float);
   size_t output_size_in_bytes = OUTPUT_SIZE * sizeof(float);


   // Allocate BOs (buffer objects) of requested size with appropriate flags
   // Memory map BOs into user's address space (DDR Memory)
   xrtBufferHandle in_bohdl = xrtBOAlloc(dhdl, input_size_in_bytes, 0, 0);
   auto in_bomapped = reinterpret_cast<float*>(xrtBOMap(in_bohdl));
   memcpy(in_bomapped, input_pl, input_size_in_bytes); // set input mapped region to same data as input pl
   printf("Input memory virtual addr 0x%p\n", in_bomapped);

   xrtBufferHandle out_bohdl = xrtBOAlloc(dhdl, output_size_in_bytes, 0, 0);
   auto out_bomapped = reinterpret_cast<float*>(xrtBOMap(out_bohdl));
   printf("Output memory virtual addr 0x%p\n", out_bomapped);

   
   // Read in data from file
   // std::ifstream inp_file;
   // inp_file.open("data/concat_in.txt", std::ifstream::in);

   #ifdef __SYNCB0_ENABLE__
      xrtBOSync(in_bohdl, XCL_BO_SYNC_BO_TO_DEVICE, input_size_in_bytes, 0);
      printf("xrtBOSync done.\n")
   #endif


   // Create kernel handle, runtime handle, set args, start kernels
   xrtKernelHandle s2mm_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "s2mm:{s2mm_0}");
   xrtRunHandle s2mm_rhdl = xrtRunOpen(s2mm_khdl); 
   xrtRunSetArg(s2mm_rhdl, 0, out_bohdl);
   xrtRunSetArg(s2mm_rhdl, 2, OUTPUT_SIZE);
   xrtRunStart(s2mm_rhdl);

   xrtKernelHandle mm2s_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "mm2s:{mm2s_0}");
   xrtRunHandle mm2s_rhdl = xrtRunOpen(mm2s_khdl); 
   xrtRunSetArg(mm2s_rhdl, 0, in_bohdl);
   xrtRunSetArg(mm2s_rhdl, 2, INPUT_SIZE);
   xrtRunStart(mm2s_rhdl);
#endif
   
   // Graph execution for AIE
   try {
      adfCheck(fpscalar.init(), "init fpscalar");
      adfCheck(fpscalar.run(ITER_CNT), "run fpscalar");
      adfCheck(fpscalar.end(), "end fpscalar");
      // get_graph_throughput_by_port(fpscalar, "plout[0]", fpscalar.plout[0], 1*8, sizeof(float_t), ITER_CNT);
   }
   catch (const std::system_error& ex) {
      xrt::error error(deviceIdx, XRT_ERROR_CLASS_AIE);
      auto errCode = error.get_error_code();
      auto timestamp = error.get_timestamp();
      auto err_str = error.to_string();
      std::cout << timestamp << " error code:" << errCode << " Error:" << err_str << std::endl;
   }
   
#ifndef EXTERNAL_IO
   
   // Wait for Kernel execution to end, close runtime and kernel handlers
   printf("Waiting for dma hls to complete...\n");
   
   auto s2mm_state = xrtRunWait(s2mm_rhdl);
   printf("s2mm completed with status (%d)\n", s2mm_state);
   xrtRunClose(s2mm_rhdl); // xrtRunOpen
   xrtKernelClose(s2mm_khdl); // xrtPLKernelOpen
   
   auto mm2s_state = xrtRunWait(mm2s_rhdl);
   printf("mm2s completed with status (%d)\n", mm2s_state);
   xrtRunClose(mm2s_rhdl);
   xrtKernelClose(mm2s_khdl);
   
   printf("Closed runtime handlers and kernel handlers...\n");

   #ifdef __SYNCB0_ENABLE__
      xrtBOSync(out_bohdl, XCL_BO_SYNC_BO_FROM_DEVICE, output_size_in_bytes, 0);
   #endif
   
   
   // Check outputs
   int errCnt = 0;
   int errFlag = 0;
   for (int i = 0; i < OUTPUT_SIZE; i++) {
      if (out_bomapped[i] != in_bomapped[i]) {
         errFlag = errFlag || 1;
         ++errCnt;
         printf("Error: out_bomapped[%d] %f != in_bomapped[%d] %f\n", 
            i, out_bomapped[i], i, in_bomapped[i]);
      }
   }
   printf("\n");

   
   //Release allocated resources
   xrtBOFree(in_bohdl);
   xrtBOFree(out_bohdl);
   printf("Released I/O buffer objects.\n");
#endif
   
   xrtDeviceClose(dhdl);
   
#ifndef EXTERNAL_IO
   std::cout << "TEST " << (errCnt ? "FAILED" : "PASSED") << std::endl;
   return (errCnt ? EXIT_FAILURE :  EXIT_SUCCESS);
#else
   return 0;
#endif
   
}
