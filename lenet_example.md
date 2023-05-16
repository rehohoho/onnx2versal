# Lenet Example

## Generate
```
rm ../data/lenet_mnist__* ../data/mnist_test_*
cd python/
python generate_node_data.py
```

## Manually write
```
design/aie_src/graph_[net_name].h
design/aie_src/graph_[net_name].cpp
design/host_app_src/[net_name]_aie_app.cpp
design/system_configs/[net_name]_logprofile.cfg
design/system_configs/[net_name].cfg
design/traffic_gen/xtg_[net_name]_logprofile.cfg
design/traffic_gen/xtg_[net_name].cfg
```

## Make notes
```
EXTIO: 0|1        whether to use external traffic generator, only graph tests
                  host app in system tests is equivalent of traffic generator
LOG_PROFILE: 0|1  whether to stdout profile logs and output intermediates
```

## Run x86sim graph (~1 min compile)
For graph test, ITER_CNT (default 1) is used in compilation
* Analyze graph compilation after compile: <br />
`vitis_analyzer build/lenet/sw_emu/Work/graph_lenet.aiecompile_summary`
* Clean build: <br />
`TARGET=sw_emu GRAPH=lenet make clean`


<details><summary>
<b>Compile-time file input, output intermediate results:</b><br />
<code>TARGET=sw_emu EXTIO=0 LOG_PROFILE=1 GRAPH=lenet make clean_reports graph aiesim</code>
</summary>

```
stdout:
Checking 1/6: lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 2/6: lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 3/6: lenet_mnist__4___relu2_Relu___relu2_Relu_output_0__1x16x8x8.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 4/6: lenet_mnist__19___relu5_Relu__output__1x10.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 5/6: lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x6x12x12.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 6/6: lenet_mnist__1___relu1_Relu___relu1_Relu_output_0__1x6x24x24.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
```

```
files:
reports_dir/
└── lenet
    └── sw_emu
        └── x86simulator_output
            ├── lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt
            ├── lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt
            ├── lenet_mnist__19___relu5_Relu__output__1x10.txt
            ├── lenet_mnist__1___relu1_Relu___relu1_Relu_output_0__1x6x24x24.txt
            ├── lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x6x12x12.txt
            ├── lenet_mnist__4___relu2_Relu___relu2_Relu_output_0__1x16x8x8.txt
            ├── lenet_out.txt
            ├── x86sim.aierun_summary
            ├── x86sim_check.log
            └── x86simulator.log
```
</details>


<details><summary>
<b>External traffic generator, output prediction only:</b><br />
<code>TARGET=sw_emu EXTIO=1 LOG_PROFILE=0 GRAPH=lenet make clean_reports graph aiesim ITER_CNT=100</code>
</summary>

```
stdout:
17:05:09: Creating ipc_axis_master_util for plin0_lenet_input...
17:05:09: Creating ipc_axis_slave_util for plout0_lenet_argm19...
17:05:09: Begin run...
17:05:09: Running master plin0_lenet_input
17:05:09: Running slave plout0_lenet_argm19
17:05:09: [plin0_lenet_input]: Sending 39200 float32 data...
17:05:10: [plout0_lenet_argm19]: read lines #0
17:05:11: [plout0_lenet_argm19]: read lines #1
17:05:12: [plout0_lenet_argm19]: read lines #2
...
17:06:46: [plout0_lenet_argm19]: read lines #98
17:06:47: [plout0_lenet_argm19]: read lines #99
17:06:47: Slave plout0_lenet_argm19 finished. Written to /home/ruien/workspace/onnx2versal/reports_dir/lenet/sw_emu/x86simulator_output/mnist_test_label.txt
17:06:47: Master plin0_lenet_input finished.
|INFO | plout0_lenet_argm19 will connect to :: /tmp/unix_ruien:plout0_lenet_argm19
|INFO | plin0_lenet_input will connect to :: /tmp/unix_ruien:plin0_lenet_input
[init lenet]: success
[run lenet]: success
|ERROR| plin0_lenet_input Failed to read data : Address already in use1|ERROR| plout0_lenet_argm19 Failed to read data : Success1|ERROR| plout0_lenet_argm19INFO: Connection with 'plin0_lenet_input' closed.
 caught exception : Failed to read data1[end lenet]: success
Simulation completed successfully returning zero
Checking directories /home/ruien/workspace/onnx2versal/reports_dir/lenet/sw_emu/x86simulator_output and /home/ruien/workspace/onnx2versal/data
Checking 1/1: mnist_test_label.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
```

```
files:
reports_dir/
└── lenet
    └── sw_emu
        └── x86simulator_output
            ├── mnist_test_label.txt
            ├── x86sim.aierun_summary
            ├── x86sim_check.log
            ├── x86sim_filetraffic.log
            └── x86simulator.log
```
</details>


<details><summary>
<b>External traffic generator, output intermediate results:</b><br />
<code>TARGET=sw_emu EXTIO=1 LOG_PROFILE=1 GRAPH=lenet make clean_reports graph aiesim</code>
</summary>

```
stdout:
Checking 1/6: lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x6x12x12.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 2/6: lenet_mnist__4___relu2_Relu___relu2_Relu_output_0__1x16x8x8.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 3/6: lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 4/6: lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 5/6: lenet_mnist__19___relu5_Relu__output__1x10.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 6/6: lenet_mnist__1___relu1_Relu___relu1_Relu_output_0__1x6x24x24.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
```

```
files:
reports_dir/
└── lenet
    └── sw_emu
        └── x86simulator_output
            ├── lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt
            ├── lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt
            ├── lenet_mnist__19___relu5_Relu__output__1x10.txt
            ├── lenet_mnist__1___relu1_Relu___relu1_Relu_output_0__1x6x24x24.txt
            ├── lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x6x12x12.txt
            ├── lenet_mnist__4___relu2_Relu___relu2_Relu_output_0__1x16x8x8.txt
            ├── lenet_out.txt
            ├── x86sim.aierun_summary
            ├── x86sim_check.log
            ├── x86sim_filetraffic.log
            └── x86simulator.log
```
</details>
<br />



## Run x86sim system (~2 min compile)
For system test, ITER_CNT is an argument into host app.
* Analyze system compilation after compile: <br /> 
`vitis_analyzer build/lenet/sw_emu/vck190_lenet_aie.xsa.link_summary`
* Clean build: <br />
`TARGET=sw_emu GRAPH=lenet make clean`


<details><summary>
<b>Test system with x86 AIE, sysC PL, x86 PS, prediction only:</b><br />
<code>TARGET=sw_emu EXTIO=0 LOG_PROFILE=0 GRAPH=lenet make graph kernels xsa application package</code><br />
<code>TARGET=sw_emu EXTIO=0 LOG_PROFILE=0 GRAPH=lenet make clean_reports run_emu ITER_CNT=100</code>
</summary>

```
stdout:
CRITICAL WARNING: [SW-EM 09-0] Unable to find emconfig.json. Using default device "xilinx:pcie-hw-em:7v3:1.0"
CRITICAL WARNING: [SW-EM 09-0] Unable to find emconfig.json. Using default device "xilinx:pcie-hw-em:7v3:1.0"
Input memory virtual addr 0x0x7f0bc8000000
Output memory virtual addr 0x0x7f07c8000000
xrtBOSync done.
[init lenet]: success
Kernel Name: mm2s_0, CU Number: 0, Thread creation status: success
Kernel Name: s2mm_0, CU Number: 1, Thread creation status: success
Kernel Name: mm2s_0, CU Number: 0, State: Start
Kernel Name: mm2s_0, CU Number: 0, State: Running
Kernel Name: s2mm_0, CU Number: 1, State: Start
Kernel Name: s2mm_0, CU Number: 1, State: Running
Kernel Name: mm2s_0, CU Number: 0, State: Idle
Kernel Name: s2mm_0, CU Number: 1, State: Idle
device process sw_emu_device done
Kernel Name: mm2s_0, CU Number: 0, Status: Shutdown
Kernel Name: s2mm_0, CU Number: 1, Status: Shutdown
INFO [HLS SIM]: The maximum depth reached by any hls::stream() instance in the design is 76831
INFO [HLS SIM]: The maximum depth reached by any hls::stream() instance in the design is 1
[run lenet]: success
[wait lenet]: success
[end lenet]: success
Waiting for dma hls to complete...
mm2s completed with status (4)
s2mm completed with status (4)
Closed runtime handlers and kernel handlers...
Released I/O buffer objects.
Checking directories /home/ruien/workspace/onnx2versal/reports_dir/lenet/sw_emu/x86simulator_output and /home/ruien/workspace/onnx2versal/data
Checking 1/1: mnist_test_label.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
```

```
files:
reports_dir/
└── lenet
    └── sw_emu
        └── x86simulator_output
            ├── embedded_run.log
            ├── mnist_test_label.txt
            └── x86sim_check.log
```
</details>

<details><summary>
<b>Test system with x86 AIE, sysC PL, x86 PS, output intermediates:</b><br />
This fails mnist_test_label.txt check since only 1 iteration is done, should pass intermediate tests<br />
<code>TARGET=sw_emu EXTIO=0 LOG_PROFILE=1 GRAPH=lenet make graph kernels xsa application package</code><br />
<code>TARGET=sw_emu EXTIO=0 LOG_PROFILE=1 GRAPH=lenet make clean_reports run_emu ITER_CNT=1</code>
</summary>

```
stdout:
Checking 1/7: lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 2/7: lenet_mnist__19___relu5_Relu__output__1x10.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 3/7: lenet_mnist__1___relu1_Relu___relu1_Relu_output_0__1x6x24x24.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 4/7: mnist_test_label.txt
WARNING: arr1 shape (1, 2), arr2 shape (50, 2)
TEST (tolerance): first 2 FAILED! (rtol=1e-03, atol=1e-05)
Checking 5/7: lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x6x12x12.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 6/7: lenet_mnist__4___relu2_Relu___relu2_Relu_output_0__1x16x8x8.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 7/7: lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Traceback (most recent call last):
  File "/home/ruien/workspace/onnx2versal/check.py", line 74, in <module>
    assert(pass_count == len(filenames)), f"{pass_count} / {len(filenames)} tests passed."
AssertionError: 6 / 7 tests passed.
```

```
files:
└── lenet
    └── sw_emu
        └── x86simulator_output
            ├── embedded_run.log
            ├── lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt
            ├── lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt
            ├── lenet_mnist__19___relu5_Relu__output__1x10.txt
            ├── lenet_mnist__1___relu1_Relu___relu1_Relu_output_0__1x6x24x24.txt
            ├── lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x6x12x12.txt
            ├── lenet_mnist__4___relu2_Relu___relu2_Relu_output_0__1x16x8x8.txt
            ├── mnist_test_label.txt
            └── x86sim_check.log
```
</details>
<br />


## Run sysC graph (~4.5 min compile, ~8 min run)
Only can output maximum of 6-7 intermediates since AIE only has 8 cascade lines. aiesim_profile allows printf and timing.
* Analyze graph compilation after compile: <br />
`vitis_analyzer build/lenet/hw_emu/Work/graph_lenet.aiecompile_summary`
* Analyze profiled info after run (includes graph analysis): <br />
`vitis_analyzer reports_dir/lenet/hw_emu/aiesimulator_output/default.aierun_summary`
* Clean build: <br />
`TARGET=hw_emu GRAPH=lenet make clean`


<details><summary>
<b>Test the graph with SysC, outputs intermediate results:</b><br />
<code>TARGET=hw_emu EXTIO=0 LOG_PROFILE=1 GRAPH=lenet make graph clean_reports aiesim_profile</code>
</summary>

```
stdout:
[init lenet]: success
Set iterations for the core(s) of graph lenet
Enabling core(s) of graph lenet
[run graph]: success
Waiting for core(s) of graph lenet to finish execution ...
Running Conv5x5ReluBCHW<28, 24, 1, 1, 6>
start = 3647,end = 24918,total = 21271
Running Maxpool2x2BCHW::filter<24, 12, 1, 6>
start = 29335,end = 30236,total = 901
Running Conv5x5ReluBCHW<12, 8, 1, 6, 16>
start = 31414,end = 72938,total = 41524
Running Maxpool2x2BCHW::filter<8, 4, 1, 16>
start = 74309,end = 74600,total = 291
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
start = 75024,end = 75927,total = 903
start = 75028,end = 75931,total = 903
start = 75028,end = 75931,total = 903
start = 75032,end = 75935,total = 903
start = 75036,end = 75939,total = 903
start = 75040,end = 75943,total = 903
start = 75044,end = 75947,total = 903
start = 75048,end = 75951,total = 903
Running Concat8Scalar<16, 16, 120>
start = 76162,end = 76724,total = 562
Running GemmReluMKKN<1, 120, 32>
Running GemmReluMKKN<1, 120, 32>
Running GemmReluMKKN<1, 120, 32>
start = 77040,end = 77928,total = 888
start = 77044,end = 77932,total = 888
start = 77048,end = 77936,total = 888
Running Concat3Scalar<32, 32, 84>
start = 78131,end = 78962,total = 831
Running GemmReluMKKN<1, 84, 48>
start = 79226,end = 80199,total = 973
Running Concat1Scalar<48, 48, 10>
start = 80343,end = 81221,total = 878
Running ArgmaxScalar<10, 10>
start = 81407,end = 81543,total = 136
core(s) are done executing
[wait graph]: success
plout[0]: Cycles 21 Throughput 476190476.190476 samples/s
Waiting for core(s) of graph lenet to finish execution ...
core(s) are done executing
[end lenet]: success
Exiting!
generate profile data for all cores
generate profile data for all cores
Stopping Simulator.
...
Checking 1/6: lenet_mnist__1___relu1_Relu___relu1_Relu_output_0__1x6x24x24.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 2/6: lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 3/6: lenet_mnist__19___relu5_Relu__output__1x10.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 4/6: lenet_mnist__4___relu2_Relu___relu2_Relu_output_0__1x16x8x8.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 5/6: lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 6/6: lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x6x12x12.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
```

```
files:
reports_dir/
└── lenet
    ├── hw_emu
    │   └── aiesimulator_output
    │       ├── aiesim.log
    │       ├── aiesim_options.txt
    │       ├── default.aierun_summary
    │       ├── lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt
    │       ├── lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt
    │       ├── lenet_mnist__19___relu5_Relu__output__1x10.txt
    │       ├── lenet_mnist__1___relu1_Relu___relu1_Relu_output_0__1x6x24x24.txt
    │       ├── lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x6x12x12.txt
    │       ├── lenet_mnist__4___relu2_Relu___relu2_Relu_output_0__1x16x8x8.txt
    │       ├── lenet_out.txt
    │       ├── memconfig.json
    │       ├── profile_funct_22_4.txt
    │       ├── profile_funct_22_4.xml
    │       ├── profile_funct_22_5.txt
    │       ├── profile_funct_22_5.xml
    │       ...
    │       ├── profile_instr_28_6.txt
    │       ├── profile_instr_28_6.xml
    │       ├── region_0x0_0x7fffffff.mmap.dat
    │       └── x86sim_check.log
```
</details>

<details><summary>
<b>Test the graph with SysC, outputs predictions, use external traffic generator:</b><br />
<code>TARGET=hw_emu EXTIO=1 LOG_PROFILE=0 GRAPH=lenet make graph clean_reports aiesim_profile</code>
</summary>

```
stdout:
```

```
files:
```
</details>
<br />


<details><summary>
<b>Test the graph with SysC, outputs intermediate result, use external traffic generator:</b><br />
<code>TARGET=hw_emu EXTIO=1 LOG_PROFILE=1 GRAPH=lenet make graph clean_reports aiesim_profile</code>
</summary>

```
stdout:
[INFO]: Enabled Stream Switch Port Latency 
18:04:39: Creating ipc_axis_master_util for plin0_lenet_input...
18:04:39: Creating ipc_axis_slave_util for plout0_lenet_argm19...
IP-INFO: [ps_i28_ps_main] IP started.
18:04:40: Creating ipc_axis_slave_util for plout1_lenet_conv00...
18:04:40: Creating ipc_axis_slave_util for plout2_lenet_pool01...
18:04:40: Creating ipc_axis_slave_util for plout3_lenet_conv02...
18:04:40: Creating ipc_axis_slave_util for plout4_lenet_gemm14...
18:04:40: Creating ipc_axis_slave_util for plout5_lenet_gemm16...
18:04:40: Creating ipc_axis_slave_util for plout6_lenet_gemm18...
18:04:40: Begin run...
18:04:40: Running master plin0_lenet_input
18:04:40: [plin0_lenet_input]: Sending 392 float32 data...
18:04:40: Running slave plout0_lenet_argm19
18:04:40: Running slave plout1_lenet_conv00
18:04:40: Running slave plout2_lenet_pool01
18:04:40: Running slave plout3_lenet_conv02
18:04:40: Running slave plout4_lenet_gemm14
18:04:40: Running slave plout5_lenet_gemm16
18:04:40: Running slave plout6_lenet_gemm18
...
[init lenet]: success
Set iterations for the core(s) of graph lenet
Enabling core(s) of graph lenet
[run graph]: success
Waiting for core(s) of graph lenet to finish execution ...
Running Conv5x5ReluBCHW<28, 24, 1, 1, 6>
start = 3462,end = 24733,total = 21271
Running Maxpool2x2BCHW::filter<24, 12, 1, 6>
18:08:45: [plout1_lenet_conv00]: read lines #0
start = 28357,end = 29258,total = 901
Running Conv5x5ReluBCHW<12, 8, 1, 6, 16>
18:09:01: [plout2_lenet_pool01]: read lines #0
start = 30297,end = 71821,total = 41524
Running Maxpool2x2BCHW::filter<8, 4, 1, 16>
18:15:00: [plout3_lenet_conv02]: read lines #0
start = 73009,end = 73300,total = 291
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
start = 73724,end = 74627,total = 903
start = 73728,end = 74631,total = 903
start = 73728,end = 74631,total = 903
start = 73732,end = 74635,total = 903
start = 73736,end = 74639,total = 903
start = 73740,end = 74643,total = 903
start = 73744,end = 74647,total = 903
start = 73748,end = 74651,total = 903
Running Concat8Scalar<16, 16, 120>
start = 74862,end = 75424,total = 562
Running GemmReluMKKN<1, 120, 32>
Running GemmReluMKKN<1, 120, 32>
Running GemmReluMKKN<1, 120, 32>
18:15:23: [plout4_lenet_gemm14]: read lines #0
start = 75740,end = 76628,total = 888
start = 75744,end = 76632,total = 888
start = 75748,end = 76636,total = 888
Running Concat3Scalar<32, 32, 84>
start = 76831,end = 77662,total = 831
Running GemmReluMKKN<1, 84, 48>
18:15:42: [plout5_lenet_gemm16]: read lines #0
start = 77926,end = 78899,total = 973
Running Concat1Scalar<48, 48, 10>
start = 79043,end = 79921,total = 878
Running ArgmaxScalar<10, 10>
18:16:00: [plout6_lenet_gemm18]: read lines #0
start = 80107,end = 80243,total = 136
18:16:02: [plout0_lenet_argm19]: read lines #0
18:16:02: Slave plout0_lenet_argm19 finished. Written to /home/ruien/workspace/onnx2versal/reports_dir/lenet/hw_emu/aiesimulator_output/lenet_out.txt
18:16:02: Slave plout1_lenet_conv00 finished. Written to /home/ruien/workspace/onnx2versal/reports_dir/lenet/hw_emu/aiesimulator_output/lenet_mnist__1___relu1_Relu___relu1_Relu_output_0__1x6x24x24.txt
18:16:02: Slave plout2_lenet_pool01 finished. Written to /home/ruien/workspace/onnx2versal/reports_dir/lenet/hw_emu/aiesimulator_output/lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x6x12x12.txt
18:16:02: Slave plout3_lenet_conv02 finished. Written to /home/ruien/workspace/onnx2versal/reports_dir/lenet/hw_emu/aiesimulator_output/lenet_mnist__4___relu2_Relu___relu2_Relu_output_0__1x16x8x8.txt
18:16:02: Slave plout4_lenet_gemm14 finished. Written to /home/ruien/workspace/onnx2versal/reports_dir/lenet/hw_emu/aiesimulator_output/lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt
18:16:02: Slave plout5_lenet_gemm16 finished. Written to /home/ruien/workspace/onnx2versal/reports_dir/lenet/hw_emu/aiesimulator_output/lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt
18:16:02: Slave plout6_lenet_gemm18 finished. Written to /home/ruien/workspace/onnx2versal/reports_dir/lenet/hw_emu/aiesimulator_output/lenet_mnist__19___relu5_Relu__output__1x10.txt
18:16:02: Master plin0_lenet_input finished.
...
Checking 1/6: lenet_mnist__1___relu1_Relu___relu1_Relu_output_0__1x6x24x24.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 2/6: lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 3/6: lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 4/6: lenet_mnist__4___relu2_Relu___relu2_Relu_output_0__1x16x8x8.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 5/6: lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x6x12x12.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 6/6: lenet_mnist__19___relu5_Relu__output__1x10.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
```

```
files:
reports_dir/
└── lenet
    ├── hw_emu
    │   └── aiesimulator_output
    │       ├── aiesim_filetraffic.log
    │       ├── aiesim.log
    │       ├── aiesim_options.txt
    │       ├── default.aierun_summary
    │       ├── lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt
    │       ├── lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt
    │       ├── lenet_mnist__19___relu5_Relu__output__1x10.txt
    │       ├── lenet_mnist__1___relu1_Relu___relu1_Relu_output_0__1x6x24x24.txt
    │       ├── lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x6x12x12.txt
    │       ├── lenet_mnist__4___relu2_Relu___relu2_Relu_output_0__1x16x8x8.txt
    │       ├── lenet_out.txt
    │       ├── memconfig.json
    │       ├── profile_funct_22_4.txt
    │       ├── profile_funct_22_4.xml
    │       ├── profile_funct_22_5.txt
    │       ├── profile_funct_22_5.xml
    │       ...
    │       ├── profile_instr_28_6.txt
    │       ├── profile_instr_28_6.xml
    │       ├── region_0x0_0x7fffffff.mmap.dat
    │       └── x86sim_check.log
```
</details>
<br />



## Run sysC system (~15min compile, ~30min run)
* Analyze system compilation after compile: <br />
`vitis_analyzer build/lenet/hw_emu/vck190_lenet_aie.xsa.link_summary`
* Analyze profiled info after run (includes system analysis): <br />
`vitis_analyzer reports_dir/lenet/hw_emu/aiesimulator_output/default.aierun_summary`
* Clean build: <br />
`TARGET=hw_emu GRAPH=lenet make clean`


<details><summary>
<b>Test system with SysC AIE, SysC PL, SysC NoC, SysC DDR, QEMU PS:</b><br />
This fails mnist_test_label.txt check since only 1 iteration is done, should pass intermediate tests<br />
<code>TARGET=hw_emu EXTIO=0 LOG_PROFILE=0 GRAPH=lenet make clean graph kernels xsa application package</code><br />
<code>TARGET=hw_emu EXTIO=0 LOG_PROFILE=0 GRAPH=lenet make clean_reports run_emu ITER_CNT=1</code>
</summary>

```
stdout:
Initializing ADF API...

Config:
xclbin: a.xclbin
iter_cnt: 1
data_dir: data/
out_dir: output/

XAIEFAL: INFO: Resource group Avail is created.
XAIEFAL: INFO: Resource group Static is created.
XAIEFAL: INFO: Resource group Generic is created.
Input memory virtual addr 0x0xffffa50f5000
Output memory virtual addr 0x0xffffa50f4000
Inter1 memory virtual addr 0x0xffff85eac000
Inter2 memory virtual addr 0x0xffff85eab000
Inter3 memory virtual addr 0x0xffff85eaa000
Inter4 memory virtual addr 0x0xffff85ea9000
Inter5 memory virtual addr 0x0xffff85ea8000
Inter6 memory virtual addr 0x0xffff85ea7000
[init lenet]: success
XRT build version: 2.14.0
Build hash: 43926231f7183688add2dccfd391b36a1f000bea
Build date: 2022-10-07 05:12:02
Git branch: 2022.2
PID: 592
UID: 0
[Mon May 15 01:37:55 2023 GMT]
HOST: 
EXE: /mnt/lenet_aie_xrt.elf
[XRT] ERROR: Can't start profiling: port name 'plout0_lenet_argm19' not found: Invalid argument
[run graph]: success
[wait graph]: success
plout[0]: ERROR: Invalid handle. Only two performance counter in a AIE-PL interface tile. Event profile is not supported for x86sim.
[end lenet]: success
Waiting for dma hls to complete...
mm2s completed with status (4)
s2mm completed with status (4)
Closed runtime handlers and kernel handlers...
inter1 completed with status (4)
inter2 completed with status (4)
inter3 completed with status (4)
inter4 completed with status (4)
inter5 completed with status (4)
inter6 completed with status (4)
Released I/O buffer objects.
INFO: TEST PASSED, RC=0

11 minutes and 3 seconds elapsed.

Checking 1/7: lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 2/7: mnist_test_label.txt
WARNING: arr1 shape (50, 2), arr2 shape (1, 2)
TEST (tolerance): first 2 FAILED! (rtol=1e-03, atol=1e-05)
Checking 3/7: lenet_mnist__1___relu1_Relu___relu1_Relu_output_0__1x6x24x24.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 4/7: lenet_mnist__4___relu2_Relu___relu2_Relu_output_0__1x16x8x8.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 5/7: lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 6/7: lenet_mnist__19___relu5_Relu__output__1x10.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 7/7: lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x6x12x12.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
```

```
files:
outputs/
└── lenet_mnist__15___relu3_Relu___relu3_Relu_output_0__1x120.txt
└── lenet_mnist__17___relu4_Relu___relu4_Relu_output_0__1x84.txt
└── lenet_mnist__19___relu5_Relu__output__1x10.txt
└── lenet_mnist__1___relu1_Relu___relu1_Relu_output_0__1x6x24x24.txt
└── lenet_mnist__2___pool1_MaxPool___pool1_MaxPool_output_0__1x6x12x12.txt
└── lenet_mnist__4___relu2_Relu___relu2_Relu_output_0__1x16x8x8.txt
└── mnist_test_label.txt
```
</details>
<br />



## Test system on hardware (~1h compile, ~6min flash, ~10s boot, ~0s run)

<details><summary>
<b>Test system on hardware:</b><br />
<code>TARGET=hw EXTIO=0 LOG_PROFILE=0 GRAPH=lenet make clean graph kernels xsa application package</code><br />
<code>sudo dd if=build/lenet/hw_emu/package/sd_card.img of=/dev/mmcblk0 conv=fsync status=progress</code>
</summary>

```
stdout:
```

```
files:
outputs/
└── mnist_test_label.txt
```
</details>
<br />
