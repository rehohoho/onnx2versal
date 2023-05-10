#include <fstream>

#include "graph_lenet.cpp"

#include "experimental/xrt_aie.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_bo.h"
#include "experimental/xrt_error.h"

#include "adf/adf_api/XRTConfig.h"


#define V_PER_LINE      2
#define INPUT_FILENAME  "mnist_test_data.txt"
#define GOLDEN_FILENAME "mnist_test_label.txt"
#define OUTPUT_FILENAME "mnist_test_label.txt"

#define INTER1_FILENAME "lenet_mnist__1___relu1_Relu___relu1_Relu_output_0__1x6x24x24.txt"
#define INTER2_FILENAME "lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x6x12x12.txt"
#define INTER3_FILENAME "lenet_mnist__4___relu2_Relu___relu2_Relu_output_0__1x16x8x8.txt"
#define INTER4_FILENAME "lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt"
#define INTER5_FILENAME "lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt"
#define INTER6_FILENAME "lenet_mnist__19___relu5_Relu__output__1x10.txt"


void write_arr_to_file(
   const std::string& filename,
   const float* bomapped,
   const size_t bosize
) {
   std::ofstream file;
   file.open(filename, std::ofstream::out);
   if (!file) printf("Unable to open %s\n", filename.c_str());
   for (int j = 0; j < bosize; j+=V_PER_LINE) {
      for (int k = 0; k < V_PER_LINE; k++) {
         file << bomapped[j+k] << " ";
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

   const int input_size = 28*28*iter_cnt;
   const int output_size = 1*iter_cnt;

   // Open device, load xclbin
   auto deviceIdx = xrt::device(0);
   auto dhdl = xrtDeviceOpen(0);
   auto xclbin = load_xclbin(dhdl, xclbin_path);
   auto top = reinterpret_cast<const axlf*>(xclbin.data());


#ifndef EXTERNAL_IO
   // Allocate BOs (buffer objects) of requested size with appropriate flags
   // Memory map BOs into user's address space (DDR Memory)
   size_t input_size_in_bytes = input_size * sizeof(float);
   size_t output_size_in_bytes = output_size * sizeof(float);

   xrtBufferHandle in_bohdl = xrtBOAlloc(dhdl, input_size_in_bytes, 0, 0);
   xrtBufferHandle out_bohdl = xrtBOAlloc(dhdl, output_size_in_bytes, 0, 0);

   auto in_bomapped = reinterpret_cast<float*>(xrtBOMap(in_bohdl));
   auto out_bomapped = reinterpret_cast<float*>(xrtBOMap(out_bohdl));
   
   printf("Input memory virtual addr 0x%p\n", in_bomapped);
   printf("Output memory virtual addr 0x%p\n", out_bomapped);

   // Read in data from file
   std::ifstream inp_file;
   inp_file.open(data_dir+INPUT_FILENAME, std::ifstream::in);
   if (!inp_file) printf("Unable to open %s.\n", (data_dir+INPUT_FILENAME).c_str());
   float d;
   for (int j = 0; j < input_size; j+=V_PER_LINE) {
      for (int k = 0; k < V_PER_LINE; k++) {
         inp_file >> d;
         in_bomapped[j+k] = d;
      }
   }

#ifdef __SYNCB0_ENABLE__
   xrtBOSync(in_bohdl, XCL_BO_SYNC_BO_TO_DEVICE, input_size_in_bytes, 0);
   printf("xrtBOSync done.\n")
#endif


   // Create kernel handle, runtime handle, set args, start kernels
   xrtKernelHandle in_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "mm2s:{mm2s_0}");
   xrtKernelHandle out_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "s2mm:{s2mm_0}");
   xrtRunHandle in_rhdl = xrtRunOpen(in_khdl); 
   xrtRunHandle out_rhdl = xrtRunOpen(out_khdl); 
   
   xrtRunSetArg(in_rhdl, 0, in_bohdl);
   xrtRunSetArg(in_rhdl, 2, input_size);
   xrtRunSetArg(out_rhdl, 0, out_bohdl);
   xrtRunSetArg(out_rhdl, 2, output_size);

   xrtRunStart(in_rhdl);
   xrtRunStart(out_rhdl);

#ifdef DEBUG
   size_t inter1_size = 1*6*24*24;
   size_t inter2_size = 1*6*12*12;
   size_t inter3_size = 1*16*8*8;
   size_t inter4_size = 1*120;
   size_t inter5_size = 1*84;
   size_t inter6_size = 1*10;

   xrtBufferHandle inter1_bohdl = xrtBOAlloc(dhdl, inter1_size * sizeof(float), 0, 0);
   xrtBufferHandle inter2_bohdl = xrtBOAlloc(dhdl, inter2_size * sizeof(float), 0, 0);
   xrtBufferHandle inter3_bohdl = xrtBOAlloc(dhdl, inter3_size * sizeof(float), 0, 0);
   xrtBufferHandle inter4_bohdl = xrtBOAlloc(dhdl, inter4_size * sizeof(float), 0, 0);
   xrtBufferHandle inter5_bohdl = xrtBOAlloc(dhdl, inter5_size * sizeof(float), 0, 0);
   xrtBufferHandle inter6_bohdl = xrtBOAlloc(dhdl, inter6_size * sizeof(float), 0, 0);

   auto inter1_bomapped = reinterpret_cast<float*>(xrtBOMap(inter1_bohdl));
   auto inter2_bomapped = reinterpret_cast<float*>(xrtBOMap(inter2_bohdl));
   auto inter3_bomapped = reinterpret_cast<float*>(xrtBOMap(inter3_bohdl));
   auto inter4_bomapped = reinterpret_cast<float*>(xrtBOMap(inter4_bohdl));
   auto inter5_bomapped = reinterpret_cast<float*>(xrtBOMap(inter5_bohdl));
   auto inter6_bomapped = reinterpret_cast<float*>(xrtBOMap(inter6_bohdl));

   printf("Inter1 memory virtual addr 0x%p\n", inter1_bomapped);
   printf("Inter2 memory virtual addr 0x%p\n", inter2_bomapped);
   printf("Inter3 memory virtual addr 0x%p\n", inter3_bomapped);
   printf("Inter4 memory virtual addr 0x%p\n", inter4_bomapped);
   printf("Inter5 memory virtual addr 0x%p\n", inter5_bomapped);
   printf("Inter6 memory virtual addr 0x%p\n", inter6_bomapped);

   xrtKernelHandle inter1_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "s2mm:{s2mm_1}");
   xrtKernelHandle inter2_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "s2mm:{s2mm_2}");
   xrtKernelHandle inter3_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "s2mm:{s2mm_3}");
   xrtKernelHandle inter4_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "s2mm:{s2mm_4}");
   xrtKernelHandle inter5_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "s2mm:{s2mm_5}");
   xrtKernelHandle inter6_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "s2mm:{s2mm_6}");

   xrtRunHandle inter1_rhdl = xrtRunOpen(inter1_khdl);
   xrtRunHandle inter2_rhdl = xrtRunOpen(inter2_khdl);
   xrtRunHandle inter3_rhdl = xrtRunOpen(inter3_khdl);
   xrtRunHandle inter4_rhdl = xrtRunOpen(inter4_khdl);
   xrtRunHandle inter5_rhdl = xrtRunOpen(inter5_khdl);
   xrtRunHandle inter6_rhdl = xrtRunOpen(inter6_khdl);

   xrtRunSetArg(inter1_rhdl, 0, inter1_bohdl);
   xrtRunSetArg(inter1_rhdl, 2, inter1_size);
   xrtRunSetArg(inter2_rhdl, 0, inter2_bohdl);
   xrtRunSetArg(inter2_rhdl, 2, inter2_size);
   xrtRunSetArg(inter3_rhdl, 0, inter3_bohdl);
   xrtRunSetArg(inter3_rhdl, 2, inter3_size);
   xrtRunSetArg(inter4_rhdl, 0, inter4_bohdl);
   xrtRunSetArg(inter4_rhdl, 2, inter4_size);
   xrtRunSetArg(inter5_rhdl, 0, inter5_bohdl);
   xrtRunSetArg(inter5_rhdl, 2, inter5_size);
   xrtRunSetArg(inter6_rhdl, 0, inter6_bohdl);
   xrtRunSetArg(inter6_rhdl, 2, inter6_size);

   xrtRunStart(inter1_rhdl);
   xrtRunStart(inter2_rhdl);
   xrtRunStart(inter3_rhdl);
   xrtRunStart(inter4_rhdl);
   xrtRunStart(inter5_rhdl);
   xrtRunStart(inter6_rhdl);
#endif

#endif

   // Graph execution for AIE
   adf::registerXRT(dhdl, top->m_header.uuid);
   try {
      adfCheck(lenet.init(), "init lenet");
      get_graph_throughput_by_port(lenet, "plout[0]", lenet.plout[0], 1*iter_cnt, sizeof(float_t), iter_cnt);
      adfCheck(lenet.end(), "end lenet");
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
   
   auto in_state = xrtRunWait(in_rhdl);
   auto out_state = xrtRunWait(out_rhdl);
   
   printf("mm2s completed with status (%d)\n", in_state);
   printf("s2mm completed with status (%d)\n", out_state);
   
   xrtRunClose(in_rhdl); // xrtRunOpen
   xrtRunClose(out_rhdl);
   
   xrtKernelClose(in_khdl); // xrtPLKernelOpen
   xrtKernelClose(out_khdl);
   
   printf("Closed runtime handlers and kernel handlers...\n");

#ifdef __SYNCB0_ENABLE__
      xrtBOSync(out_bohdl, XCL_BO_SYNC_BO_FROM_DEVICE, output_size_in_bytes, 0);
#endif


#ifdef DEBUG
   auto inter1_state = xrtRunWait(inter1_rhdl);
   auto inter2_state = xrtRunWait(inter2_rhdl);
   auto inter3_state = xrtRunWait(inter3_rhdl);
   auto inter4_state = xrtRunWait(inter4_rhdl);
   auto inter5_state = xrtRunWait(inter5_rhdl);
   auto inter6_state = xrtRunWait(inter6_rhdl);

   printf("inter1 completed with status (%d)\n", inter1_state);
   printf("inter2 completed with status (%d)\n", inter2_state);
   printf("inter3 completed with status (%d)\n", inter3_state);
   printf("inter4 completed with status (%d)\n", inter4_state);
   printf("inter5 completed with status (%d)\n", inter5_state);
   printf("inter6 completed with status (%d)\n", inter6_state);

   xrtRunClose(inter1_rhdl);
   xrtRunClose(inter2_rhdl);
   xrtRunClose(inter3_rhdl);
   xrtRunClose(inter4_rhdl);
   xrtRunClose(inter5_rhdl);
   xrtRunClose(inter6_rhdl);
   
   xrtKernelClose(inter1_khdl);
   xrtKernelClose(inter2_khdl);
   xrtKernelClose(inter3_khdl);
   xrtKernelClose(inter4_khdl);
   xrtKernelClose(inter5_khdl);
   xrtKernelClose(inter6_khdl);

#ifdef __SYNCB0_ENABLE__
   xrtBOSync(inter1_bohdl, XCL_BO_SYNC_BO_FROM_DEVICE, inter1_size * sizeof(float), 0);
   xrtBOSync(inter2_bohdl, XCL_BO_SYNC_BO_FROM_DEVICE, inter2_size * sizeof(float), 0);
   xrtBOSync(inter3_bohdl, XCL_BO_SYNC_BO_FROM_DEVICE, inter3_size * sizeof(float), 0);
   xrtBOSync(inter4_bohdl, XCL_BO_SYNC_BO_FROM_DEVICE, inter4_size * sizeof(float), 0);
   xrtBOSync(inter5_bohdl, XCL_BO_SYNC_BO_FROM_DEVICE, inter5_size * sizeof(float), 0);
   xrtBOSync(inter6_bohdl, XCL_BO_SYNC_BO_FROM_DEVICE, inter6_size * sizeof(float), 0);
#endif

   write_arr_to_file(out_dir+INTER1_FILENAME, inter1_bomapped, inter1_size);
   write_arr_to_file(out_dir+INTER2_FILENAME, inter2_bomapped, inter2_size);
   write_arr_to_file(out_dir+INTER3_FILENAME, inter3_bomapped, inter3_size);
   write_arr_to_file(out_dir+INTER4_FILENAME, inter4_bomapped, inter4_size);
   write_arr_to_file(out_dir+INTER5_FILENAME, inter5_bomapped, inter5_size);
   write_arr_to_file(out_dir+INTER6_FILENAME, inter6_bomapped, inter6_size);

   xrtBOFree(inter1_bohdl);
   xrtBOFree(inter2_bohdl);
   xrtBOFree(inter3_bohdl);
   xrtBOFree(inter4_bohdl);
   xrtBOFree(inter5_bohdl);
   xrtBOFree(inter6_bohdl);
#endif

   
   // Write and check outputs
   std::ofstream oup_file;
   oup_file.open(out_dir+OUTPUT_FILENAME, std::ofstream::out);
   if (!oup_file) printf("Unable to open %s\n", (out_dir+OUTPUT_FILENAME).c_str());
   std::ifstream chk_file;
   chk_file.open(data_dir+GOLDEN_FILENAME, std::ifstream::in);
   if (!chk_file) printf("Unable to open %s.\n", (data_dir+GOLDEN_FILENAME).c_str());

   float g;
   int match = 0;
   for (int j = 0; j < output_size; j+=V_PER_LINE) {
      for (int k = 0; k < V_PER_LINE; k++) {
         chk_file >> g;
         if (g != out_bomapped[j+k]) 
            match = 1;
         oup_file << out_bomapped[j+k] << " ";
      }
      oup_file << std::endl;
   }

   
   //Release allocated resources
   xrtBOFree(in_bohdl);
   xrtBOFree(out_bohdl);
   printf("Released I/O buffer objects.\n");
#endif

   xrtDeviceClose(dhdl);
   
#ifndef EXTERNAL_IO
   std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
   return (match ? EXIT_FAILURE :  EXIT_SUCCESS);
#else
   return 0;
#endif
   
}
