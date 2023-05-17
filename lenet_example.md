# Lenet Example

## Requirements and setup
* See [README Requirements](README.md#Requirements)
* See [README Setup](README.md#Setup)
* Make sure you don't `source sample_env_setup.sh` as it overwrites your python environment for this section.

## Train Lenet model on MNIST
```
$ cd python/;python train.py
stdout:
accuracy: 0.xxx
Model finished training
Converted to onnx
Exported model has been tested with ONNXRuntime, and the result looks good!

files:
models/
├── lenet_mnist.onnx
└── lenet_mnist.pkl
```

## Generate files
```
$ python generate.py

stdout:
WARNING: fusing Conv+Relu
Padding Conv weights (6, 1, 5, 5) to (6, 1, 5, 8)
WARNING: fusing Conv+Relu
Padding Conv weights (16, 6, 5, 5) to (16, 6, 5, 8)
WARNING: Shape not implemented, skipping...
WARNING: Constant not implemented, skipping...
WARNING: Gather not implemented, skipping...
WARNING: Constant not implemented, skipping...
WARNING: Unsqueeze not implemented, skipping...
WARNING: Constant not implemented, skipping...
WARNING: Concat not implemented, skipping...
WARNING: Reshape not implemented, skipping...
Found matching output /Reshape_output_0 and k5pool output
Padding Gemm weights (84, 10) to (84, 12)
Generating MNIST txt for 100 data points

files:
models/
├── lenet_mnist_inter.onnx
├── lenet_mnist.onnx
└── lenet_mnist.pkl
data/
├── input.txt
├── k0conv_goldenout.txt
├── k0conv_in.txt
├── k14gemm_goldenout.txt
├── k14gemm_in.txt
├── k16gemm_goldenout.txt
├── k16gemm_in.txt
├── k18gemm_goldenout.txt
├── k18gemm_in.txt
├── k2pool_goldenout.txt
├── k2pool_in.txt
├── k3conv_goldenout.txt
├── k3conv_in.txt
├── k5pool_goldenout.txt
├── k5pool_in.txt
├── mnist_test_data.txt
└── mnist_test_label.txt
design/
├── aie_src
│   └── graph_lenet_mnist.cpp
├── host_app_src
│   └── lenet_mnist_aie_app.cpp
├── system_configs
│   ├── lenet_mnist.cfg
│   └── lenet_mnist_output_inter.cfg
└── trafficgen
    ├── xtg_lenet_mnist_output_inter.py
    └── xtg_lenet_mnist.py
```

## Make options
See [README Make Options](README.md#usage---simulation-verification-profiling-hardware-build)

## Functional check: runs graph in x86simulator 
(~1 min compile, ~0s run)<br />
For graph test, ITER_CNT is an argument into graph compilation (make graph)
```
$ TARGET=sw_emu GRAPH=lenet_mnist make graph clean_reports aiesim

stdout:
...
Checking 1/7: k5pool_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 2/7: k0conv_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 3/7: k2pool_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 4/7: k18gemm_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 5/7: k16gemm_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 6/7: k3conv_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 7/7: k14gemm_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)

files:
reports_dir/
└── lenet_mnist
    └── sw_emu
        └── x86simulator_output
            ├── k0conv_goldenout.txt
            ├── k14gemm_goldenout.txt
            ├── k16gemm_goldenout.txt
            ├── k18gemm_goldenout.txt
            ├── k2pool_goldenout.txt
            ├── k3conv_goldenout.txt
            ├── k5pool_goldenout.txt
            ├── x86sim.aierun_summary
            ├── x86sim_check.log
            └── x86simulator.log
```

<b>Traffic generator will have extra stdout and files (EXTIO=1)</b>

```
extra stdout:
17:54:35: Creating ipc_axis_master_util for plin0_lenet_mnist_input...
17:54:36: Creating ipc_axis_slave_util for plout0_lenet_mnist_k18gemm...
17:54:36: Creating ipc_axis_slave_util for plout1_lenet_mnist_k0conv...
...
17:54:36: Begin run...
17:54:36: Running master plin0_lenet_mnist_input
17:54:36: [plin0_lenet_mnist_input]: Sending 392 float32 data...
17:54:36: Running slave plout0_lenet_mnist_k18gemm
17:54:36: Running slave plout1_lenet_mnist_k0conv
...
17:54:36: [plout1_lenet_mnist_k0conv]: read lines #0
17:54:36: [plout2_lenet_mnist_k2pool]: read lines #0
...
Written to /home/ruien/workspace/test/reports_dir/lenet_mnist/sw_emu/x86simulator_output/k18gemm_goldenout.txt
17:54:37: Slave plout1_lenet_mnist_k0conv finished. Written to /home/ruien/workspace/test/reports_dir/lenet_mnist/sw_emu/x86simulator_output/k0conv_goldenout.txt
17:54:37: Slave plout2_lenet_mnist_k2pool finished. Written to /home/ruien/workspace/test/
...
17:54:37: Master plin0_lenet_mnist_input finished.
|INFO | plin0_lenet_mnist_input will connect to :: /tmp/unix_ruien:plin0_lenet_mnist_input
|INFO | plout0_lenet_mnist_k18gemm will connect to :: /tmp/unix_ruien:plout0_lenet_mnist_k18gemm
|INFO | plout3_lenet_mnist_k3conv will connect to :: /tmp/unix_ruien:plout3_lenet_mnist_k3conv
|INFO | plout1_lenet_mnist_k0conv will connect to :: /tmp/unix_ruien:plout1_lenet_mnist_k0conv
...

extra files:
reports_dir/
└── lenet_mnist
    └── sw_emu
        └── x86simulator_output
             └── x86sim_filetraffic.log
```

<b>Analyze graph compilation after compile</b>
```
vitis_analyzer build/lenet_mnist/sw_emu/Work/graph_lenet_mnist.aiecompile_summary
```
<b>Clean software graph build</b>
```
TARGET=sw_emu GRAPH=lenet_mnist make clean
```


## Functional check: runs system software emulation 
(x86 graph, sysC kernels, ~2 min compile)<br />
For system test, ITER_CNT is an argument into host app compilation (make application)
```
$ TARGET=sw_emu GRAPH=lenet_mnist make graph kernels xsa application package
$ TARGET=sw_emu GRAPH=lenet_mnist make clean_reports run_emu

stdout:
Initializing ADF API...

Config:
xclbin: a.xclbin
iter_cnt: 1
data_dir: /home/ruien/workspace/test/data/
out_dir: /home/ruien/workspace/test/reports_dir/lenet_mnist/sw_emu/x86simulator_output/

CRITICAL WARNING: [SW-EM 09-0] Unable to find emconfig.json. Using default device "xilinx:pcie-hw-em:7v3:1.0"
CRITICAL WARNING: [SW-EM 09-0] Unable to find emconfig.json. Using default device "xilinx:pcie-hw-em:7v3:1.0"
Input0 memory virtual addr 0x0x7f1b94000000
Output0 memory virtual addr 0x0x7f1394000000
Inter1 memory virtual addr 0x0x7f0b94000000
Inter2 memory virtual addr 0x0x7f0394000000
...
[init lenet_mnist]: success
[run lenet_mnist]: success
[wait lenet_mnist]: success
Kernel Name: mm2s_0, CU Number: 0, Thread creation status: success
Kernel Name: s2mm_0, CU Number: 1, Thread creation status: success
Kernel Name: s2mm_1, CU Number: 2, Thread creation status: success
...
Kernel Name: mm2s_0, CU Number: 0, State: Start
Kernel Name: mm2s_0, CU Number: 0, State: Running
Kernel Name: s2mm_0, CU Number: 1, State: Start
Kernel Name: s2mm_0, CU Number: 1, State: Running
...
Running Conv5x5on8ReluBCHW<28, 24, 1, 1, 6>
start = -1,end = -1,total = 0
Running Maxpool2x2BCHW::filter<24, 12, 1, 6>
Kernel Name: s2mm_1, CU Number: 2, State: Idle
start = -1,end = -1,total = 0
Running Conv5x5on8ReluBCHW<12, 8, 1, 6, 16>
Kernel Name: s2mm_2, CU Number: 3, State: Idle
start = -1,end = -1,total = 0
Running Maxpool2x2BCHW::filter<8, 4, 1, 16>
Kernel Name: s2mm_3, CU Number: 4, State: Idle
start = -1,end = -1,total = 0
...
device process sw_emu_device done
Kernel Name: mm2s_0, CU Number: 0, Status: Shutdown
Kernel Name: s2mm_0, CU Number: 1, Status: Shutdown
...
INFO [HLS SIM]: The maximum depth reached by any hls::stream() instance in the design is 157
INFO [HLS SIM]: The maximum depth reached by any hls::stream() instance in the design is 573
[end lenet_mnist]: success
Waiting for dma hls to complete...
mm2s completed with status (4)
s2mm completed with status (4)
Closed runtime handlers and kernel handlers...
inter1 completed with status (4)
inter2 completed with status (4)
...
Released I/O buffer objects.
Checking directories /home/ruien/workspace/test/reports_dir/lenet_mnist/sw_emu/x86simulator_output and /home/ruien/workspace/test/data
Checking 1/7: k18gemm_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 2/7: k14gemm_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 3/7: k0conv_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 4/7: k5pool_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 5/7: k2pool_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 6/7: k3conv_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 7/7: k16gemm_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)

files:
reports_dir/
└── lenet_mnist
    └── sw_emu
        └── x86simulator_output
            ├── embedded_run.log
            ├── k0conv_goldenout.txt
            ├── k14gemm_goldenout.txt
            ├── k16gemm_goldenout.txt
            ├── k18gemm_goldenout.txt
            ├── k2pool_goldenout.txt
            ├── k3conv_goldenout.txt
            ├── k5pool_goldenout.txt
            └── x86sim_check.log
```

<b>Analyze system compilation after compile</b>
```
vitis_analyzer build/lenet_mnist/sw_emu/vck190_lenet_mnist_aie.xsa.link_summary
```
<b>Clean build</b>
```
TARGET=sw_emu GRAPH=lenet_mnist make clean
```


## Functional/performance check: runs graph in aiesimulator 
(with SysC ~4.5 min compile, ~8 min run)<br />
For graph test, ITER_CNT is an argument into graph compilation (make graph)
```
$ TARGET=hw_emu GRAPH=lenet_mnist make graph
$ TARGET=hw_emu GRAPH=lenet_mnist make clean_reports aiesim_profile

stdout:
Waiting for core(s) of graph lenet_mnist to finish execution ...
Running Conv5x5on8ReluBCHW<28, 24, 1, 1, 6>
start = 3236,end = 20164,total = 16928
Running Maxpool2x2BCHW::filter<24, 12, 1, 6>
start = 24570,end = 25471,total = 901
Running Conv5x5on8ReluBCHW<12, 8, 1, 6, 16>
start = 26651,end = 52727,total = 26076
Running Maxpool2x2BCHW::filter<8, 4, 1, 16>
start = 54101,end = 54392,total = 291
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
Running GemmReluMKKN<1, 256, 16>
start = 54820,end = 55723,total = 903
start = 54824,end = 55727,total = 903
start = 54824,end = 55727,total = 903
start = 54828,end = 55731,total = 903
start = 54832,end = 55735,total = 903
start = 54836,end = 55739,total = 903
start = 54840,end = 55743,total = 903
start = 54844,end = 55747,total = 903
Running ConcatVector<8, 16, 16, 120>::filter8
start = 55947,end = 56073,total = 126
Running GemmReluMKKN<1, 120, 32>
Running GemmReluMKKN<1, 120, 32>
Running GemmReluMKKN<1, 120, 32>
start = 56393,end = 57281,total = 888
start = 56397,end = 57285,total = 888
start = 56401,end = 57289,total = 888
Running ConcatVector<3, 32, 32, 84>::filter3
start = 57437,end = 57510,total = 73
Running GemmReluMKKN<1, 84, 48>
start = 57782,end = 58755,total = 973
Running ConcatScalar<1, 48, 48, 10>::filter1
start = 58899,end = 59777,total = 878
core(s) are done executing
[wait graph]: success
plout[0]: Cycles 10 Throughput 1000000000.000000 samples/s
Waiting for core(s) of graph lenet_mnist to finish execution ...
core(s) are done executing
[end lenet_mnist]: success
Exiting!
generate profile data for all cores
generate profile data for all cores
Stopping Simulator.
Info: /OSCI/SystemC: Simulation stopped by user.
...
[INFO] : Simulation Finished, Sim result: 0
Checking directories /home/ruien/workspace/test/reports_dir/lenet_mnist/hw_emu/aiesimulator_output and /home/ruien/workspace/test/data
Checking 1/7: k14gemm_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 2/7: k16gemm_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 3/7: k18gemm_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 4/7: k3conv_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 5/7: k0conv_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 6/7: k5pool_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 7/7: k2pool_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)

files:
reports_dir/
└── lenet_mnist
    └── hw_emu
        └── aiesimulator_output
            ├── aiesim.log
            ├── aiesim_options.txt
            ├── default.aierun_summary
            ├── k0conv_goldenout.txt
            ├── k14gemm_goldenout.txt
            ├── k16gemm_goldenout.txt
            ├── k18gemm_goldenout.txt
            ├── k2pool_goldenout.txt
            ├── k3conv_goldenout.txt
            ├── k5pool_goldenout.txt
            ├── memconfig.json
            ├── profile_funct_22_0.txt
            ├── profile_funct_22_0.xml
            ├── ...
            ├── region_0x0_0x7fffffff.mmap.dat
            └── aiesim_check.log
```

<b>Traffic generator will have extra stdout and files (EXTIO=1)</b>

```
extra stdout:
INFO::[ XTLM_IPC_AXIS_MASTER ] SIM-IPC's external process can be connected to instance :  plin0_lenet_mnist_input
INFO::[ XTLM_IPC_AXIS_SLAVE ] SIM-IPC's external process can be connected to instance :  plout0_lenet_mnist_k18gemm
INFO::[ XTLM_IPC_AXIS_SLAVE ] SIM-IPC's external process can be connected to instance :  plout1_lenet_mnist_k0conv
[INFO]: Enabled Stream Switch Port Latency 
18:45:08: Creating ipc_axis_master_util for plin0_lenet_mnist_input...
18:45:08: Creating ipc_axis_slave_util for plout0_lenet_mnist_k18gemm...
IP-INFO: [ps_i27_ps_main] IP started.
18:45:09: Creating ipc_axis_slave_util for plout1_lenet_mnist_k0conv...
18:45:09: Creating ipc_axis_slave_util for plout2_lenet_mnist_k2pool...
...
18:45:09: Begin run...
18:45:09: Running master plin0_lenet_mnist_input
18:45:09: [plin0_lenet_mnist_input]: Sending 392 float32 data...
18:45:09: Running slave plout0_lenet_mnist_k18gemm
18:45:09: Running slave plout1_lenet_mnist_k0conv
...
18:47:35: [plout1_lenet_mnist_k0conv]: read lines #0
18:47:46: [plout2_lenet_mnist_k2pool]: read lines #0
...
18:51:01: Slave plout0_lenet_mnist_k18gemm finished. Written to /home/ruien/workspace/test/reports_dir/lenet_mnist/hw_emu/aiesimulator_output/k18gemm_goldenout.txt
18:51:01: Slave plout1_lenet_mnist_k0conv finished. Written to /home/ruien/workspace/test/reports_dir/lenet_mnist/hw_emu/aiesimulator_output/k0conv_goldenout.txt
18:51:01: Slave plout2_lenet_mnist_k2pool finished. Written to /home/ruien/workspace/test/reports_dir/lenet_mnist/hw_emu/aiesimulator_output/k2pool_goldenout.txt
...
18:51:01: Master plin0_lenet_mnist_input finished.
WARNING::[ XTLM_IPC::006 ] plin0_lenet_mnist_input Closing Socket
WARNING::[ XTLM_IPC::006 ] plout5_lenet_mnist_k14gemm Closing Socket
WARNING::[ XTLM_IPC::006 ] plout6_lenet_mnist_k16gemm Closing Socket
...

extra files:
reports_dir/
└── lenet_mnist
    └── hw_emu
        └── aiesimulator_output
            └── aiesim_filetraffic.log
```

<b>Analyze graph compilation after compile</b>
```
vitis_analyzer build/lenet_mnist/hw_emu/Work/graph_lenet_mnist.aiecompile_summary
```
<b>Analyze profiled information after run (includes graph analysis)</b>
```
vitis_analyzer reports_dir/lenet_mnist/hw_emu/aiesimulator_output/default.aierun_summary
```
<b>Clean hardware emulation build</b>
```
TARGET=hw_emu GRAPH=lenet_mnist make clean
```


## Functional/performance check: runs system in hardware emulation with QEMU
(sysC graph, kernels NoC, DDR, ~15min compile, ~30min run)<br />
For system test, ITER_CNT is an argument into host app compilation (make application)
```
$ TARGET=hw_emu GRAPH=lenet_mnist make graph kernels xsa application package
$ TARGET=hw_emu GRAPH=lenet_mnist make graph clean_reports run_emu

stdout:
Initializing ADF API...

Config:
xclbin: a.xclbin
iter_cnt: 1
data_dir: data/
out_dir: output/

[   79.139175] zocl-drm axi:zyxclmm_drm: zocl_create_client: created KDS client for pid(576), ret: 0
[   79.142194] zocl-drm axi:zyxclmm_drm: zocl_destroy_client: client exits pid(576)
[   79.155865] zocl-drm axi:zyxclmm_drm: zocl_create_client: created KDS client for pid(576), ret: 0
...
[   81.095991] [drm] found kind 29(AIE_RESOURCES)
[   81.096374] [drm] found kind 8(IP_LAYOUT)
[   81.096586] [drm] skip kind 9(DEBUG_IP_LAYOUT) return code: -22
[   81.096772] [drm] found kind 25(AIE_METADATA)
[   81.097048] [drm] found kind 7(CONNECTIVITY)
[   81.097227] [drm] found kind 6(MEM_TOPOLOGY)
[   81.098613] [drm] Memory 0 is not reserved in device tree. Will allocate memory from CMA
[   83.707507] zocl_irq_intc ZOCL_CU_INTC.2.auto: zocl_irq_intc_add: managing IRQ 46
[   83.829517] cu_drv CU.3.auto: cu_probe: CU[0] created
[   84.699804] zocl_irq_intc ZOCL_CU_INTC.2.auto: zocl_irq_intc_add: managing IRQ 47
[   84.790210] cu_drv CU.4.auto: cu_probe: CU[1] created
...
[   91.621456] cu_drv CU.3.auto:  ffff000007cd7c10 xrt_cu_intr_thread: CU[0] start
[   92.042016] cu_drv CU.4.auto:  ffff000007c6a410 xrt_cu_intr_thread: CU[1] start
...
XAIEFAL: INFO: Resource group Avail is created.
XAIEFAL: INFO: Resource group Static is created.
XAIEFAL: INFO: Resource group Generic is created.
[   94.206063] [drm] zocl_xclbin_read_axlf 9fee7350-25b2-b56d-7e20-d9e9d2b8c46f ret: 0
[   94.748985] [drm] bitstream 9fee7350-25b2-b56d-7e20-d9e9d2b8c46f locked, ref=1
[   94.754304] zocl-drm axi:zyxclmm_drm:  ffff000001b9dc10 kds_add_context: Client pid(576) add context Domain(65535) CU(0xffff) shared(true)
[   94.756977] zocl-drm axi:zyxclmm_drm:  ffff000001b9dc10 kds_del_context: Client pid(576) del context Domain(65535) CU(0xffff)
Input0 memory virtual addr 0x0xffffb714e000
[   94.769813] [drm] bitstream 9fee7350-25b2-b56d-7e20-d9e9d2b8c46f unlocked, ref=0
[   94.875906] [drm] bitstream 9fee7350-25b2-b56d-7e20-d9e9d2b8c46f locked, ref=1
[   94.876340] zocl-drm axi:zyxclmm_drm:  ffff000001b9dc10 kds_add_context: Client pid(576) add context Domain(0) CU(0x1) shared(true)
Output0 memory virtual addr 0x0xffffb714c000
[   95.539467] zocl-drm axi:zyxclmm_drm:  ffff000001b9dc10 kds_add_context: Client pid(576) add context Domain(0) CU(0x0) shared(true)
...
[init lenet_mnist]: success
XRT build version: 2.14.0
Build hash: 43926231f7183688add2dccfd391b36a1f000bea
Build date: 2022-10-07 05:12:02
Git branch: 2022.2
PID: 576
UID: 0
[Wed May 17 02:48:04 2023 GMT]
HOST: 
EXE: /mnt/aie_xrt.elf
[XRT] ERROR: Can't start profiling: port name 'plout0_lenet_mnist_k18gemm' not found: Invalid argument
[  124.882853] hrtimer: interrupt took 343510625 ns
[run graph]: success
[wait graph]: success
plout[0]: ERROR: Invalid handle. Only two performance counter in a AIE-PL interface tile. Event profile is not supported for x86sim.
[end lenet_mnist]: success
Waiting for dma hls to complete...
mm2s completed with status (4)
[  641.684981] zocl-drm axi:zyxclmm_drm:  ffff000001b9dc10 kds_del_context: Client pid(576) del context Domain(0) CU(0x1)
s2mm completed with status (4)
[  643.379135] zocl-drm axi:zyxclmm_drm:  ffff000001b9dc10 kds_del_context: Client pid(576) del context Domain(0) CU(0x0)
Closed runtime handlers and kernel handlers...
inter1 completed with status (4)
[  643.396986] zocl-drm axi:zyxclmm_drm:  ffff000001b9dc10 kds_del_context: Client pid(576) del context Domain(0) CU(0x2)
inter2 completed with status (4)
[  643.397625] zocl-drm axi:zyxclmm_drm:  ffff000001b9dc10 kds_del_context: Client pid(576) del context Domain(0) CU(0x3)
...
Released I/O buffer objects.
[  643.401090] [drm] bitstream 9fee7350-25b2-b56d-7e20-d9e9d2b8c46f unlocked, ref=0
[  643.535997] zocl-drm axi:zyxclmm_drm: zocl_destroy_client: client exits pid(576)
[  643.555318] zocl-drm axi:zyxclmm_drm: zocl_destroy_client: client exits pid(576)
Checking directories data and output
Checking 1/7: k14gemm_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 2/7: k16gemm_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 3/7: k5pool_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 4/7: k2pool_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 5/7: k3conv_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 6/7: k0conv_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 7/7: k18gemm_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
INFO: TEST PASSED, RC=0

9 minutes and 36 seconds elapsed.

INFO: Embedded host run completed.


files: (in QEMU environment)
output/
└── lenet_mnist
    ├── k0conv_goldenout.txt
    ├── k14gemm_goldenout.txt
    ├── k16gemm_goldenout.txt
    ├── k18gemm_goldenout.txt
    ├── k2pool_goldenout.txt
    ├── k3conv_goldenout.txt
    └── k5pool_goldenout.txt
```

<b>Analyze system compilation after compile</b>
```
vitis_analyzer build/lenet_mnist/hw_emu/vck190_lenet_mnist_aie.xsa.link_summary
```
<b>Clean hardware emulation build</b>
```
TARGET=hw_emu GRAPH=lenet_mnist make clean
```

## Test system on hardware 
(~1h compile, ~6min flash, ~10s boot, ~0s run)<br />
<b>Hardware: create hardware image, flash the SD card</b>
```
$ TARGET=hw GRAPH=lenet_mnist make graph kernels xsa application package
$ sudo dd if=build/lenet_mnist/hw/package/sd_card.img of=/dev/(DEVICE) conv=fsync status=progress

stdout:

files:
outputs/
└── mnist_test_label.txt
```
