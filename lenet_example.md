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

## Quantize the model (optional)
Quantizing the model improves performance at very slight expense of accuracy. Run the following to quantize the onnx model. For more details see https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html.

```
cd ../models
python -m onnxruntime.quantization.preprocess --input lenet_mnist.onnx --output lenet_mnist_infer.onnx
cd ../python
python quantize_onnx.py ../models/lenet_mnist_infer.onnx ../models/lenet_mnist_int8.onnx 
```

For the CLI lines to execute below, substitute "lenet_mnist" with "lenet_mnist_int8" for all arguments if you intend to use the quantized model.

## Generate files
```
$ python generate.py ../models/lenet_mnist.onnx
```

Truncated stdout (for float32 model)
```
Saving tensor of shape (1, 1, 28, 28) into ../data/input.txt
WARNING: fusing Conv+Relu
Padding Conv weights (6, 1, 5, 5) to (6, 1, 5, 8)
Saving tensor of shape (1, 1, 28, 28) into ../data/k0conv_in_shape1x1x28x28.txt
...
WARNING: Shape not implemented, skipping...
WARNING: Constant not implemented, skipping...
...
Found matching output /Reshape_output_0 and k5pool output
Disabled output padding, may result in choosing scalar op instead of vector.
...
Generating MNIST txt for 100 data points
```

Files (for float32 model):
```
models/
├── lenet_mnist_inter.onnx              # generated onnx model to output at all layers
├── lenet_mnist.onnx
└── lenet_mnist.pkl
data/
├── input.txt                           # generated input and output data to verify each kernel
├── input_host.txt
├── k0conv_goldenout.txt
├── k0conv_in.txt
├── k14gemm_goldenout.txt
├── k14gemm_in.txt
├── k16gemm_goldenout.txt
├── k16gemm_in.txt
├── k18gemm_goldenout.txt
├── k18gemm_goldenout_host.txt
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
│   └── graph_lenet_mnist.cpp           # aiengine computation graph for aiecompiler
├── host_app_src
│   └── lenet_mnist_aie_app.cpp         # host code to load data and run compiled graph
├── system_configs
│   ├── lenet_mnist.cfg                 # configuration for v++ linking
│   └── lenet_mnist_output_inter.cfg
└── trafficgen
    ├── xtg_lenet_mnist_output_inter.py # traffic generators for testing computation graph
    └── xtg_lenet_mnist.py
```

## Make options
See [README Make Options](README.md#usage---simulation-verification-profiling-hardware-build)

## Functional verification
`check.py` compares two directories and does a `numpy.assert.allclose` on matching filenames.

## Performance verification
`TARGET=hw_emu` with `DLOG=1` (default) will output cycles required for each kernel. `TARGET=hw` with `EN_TRACE=1` (not default) allows event tracing and timing. 

## Functional check: runs graph in x86simulator 
For graph test, ITER_CNT is an argument into graph compilation (make graph). Note x86simulator does not simulate hardware platform, outputs will not contain cycle information. The following tests a single sample and verifies all intermediate outputs. It takes ~1min to compile and runs quickly.
```
$ TARGET=sw_emu GRAPH=lenet_mnist make graph clean_reports aiesim 
```

Truncated stdout (for float32 model)
```
...
Running Conv5x5on8ReluBCHW<28, 24, 1, 1, 6>
start = -1,end = -1,total = 0
Running Maxpool2x2FloatBCHW::filter<24,24,12,12,1,6>
start = -1,end = -1,total = 0
...
Checking 1/7: k5pool_goldenout.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
Checking 2/7: k0conv_goldenout.txt
...

```

Files (for float32 model):
```
reports_dir/
└── lenet_mnist
    └── sw_emu
        └── x86simulator_output
            ├── k0conv_goldenout.txt    # output files from computation graph
            ├── k14gemm_goldenout.txt
            ├── k16gemm_goldenout.txt
            ├── k18gemm_goldenout.txt
            ├── k2pool_goldenout.txt
            ├── k3conv_goldenout.txt
            ├── k5pool_goldenout.txt
            ├── x86sim.aierun_summary   # logs for vitis_analyzer
            ├── x86sim_check.log        # logs from python verification script
            └── x86simulator.log        # logs from x86simulator
```

<b>Traffic generator for end-to-end test with more samples.</b><br />
The following tests 100 samples and verifies final output only. Excess logs is silenced.
```
$ TARGET=sw_emu GRAPH=lenet_mnist make graph clean_reports aiesim EXTIO=1 DOUT=0 DLOG=0 ITER_CNT=100
```

Truncated stdout (for float32 model)
```
14:43:59: Creating ipc_axis_master_util for plin0_lenet_mnist_input...
14:43:59: Creating ipc_axis_slave_util for plout0_lenet_mnist_k18gemm...
14:43:59: Begin run...
14:43:59: Running master plin0_lenet_mnist_input
14:43:59: Running slave plout0_lenet_mnist_k18gemm
14:43:59: [plin0_lenet_mnist_input]: Sending 39200 float32 data...
14:44:00: [plout0_lenet_mnist_k18gemm]: read lines #0
14:44:01: [plout0_lenet_mnist_k18gemm]: read lines #1
...
14:45:25: Slave plout0_lenet_mnist_k18gemm finished. Written to /home/ruien/workspace/onnx2versal/reports_dir/lenet_mnist/sw_emu/x86simulator_output/k18gemm_goldenout_host.txt
14:45:25: Master plin0_lenet_mnist_input finished.
|INFO | plin0_lenet_mnist_input will connect to :: /tmp/unix_ruien:plin0_lenet_mnist_input
|INFO | plout0_lenet_mnist_k18gemm will connect to :: /tmp/unix_ruien:plout0_lenet_mnist_k18gemm
[init lenet_mnist]: success
[run lenet_mnist]: success
INFO: Connection with 'plin0_lenet_mnist_input' closed.
[end lenet_mnist]: success
Simulation completed successfully returning zero
Checking directories /home/ruien/workspace/onnx2versal/reports_dir/lenet_mnist/sw_emu/x86simulator_output and /home/ruien/workspace/onnx2versal/data
Checking 1/1: k18gemm_goldenout_host.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)

```
Files (for float32 model):
```
reports_dir/
└── lenet_mnist
    └── sw_emu
        └── x86simulator_output
             └── x86sim_filetraffic.log # logs from python traffic generator script
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
For system test, ITER_CNT is an argument into host app compilation (make application). Note this uses x86simulator files, outputs will not contain cycle information. It takes ~2 min to compile and runs quickly.
```
# test 1 sample, verifies intermediates
$ TARGET=sw_emu GRAPH=lenet_mnist make graph kernels xsa application package
$ TARGET=sw_emu GRAPH=lenet_mnist make clean_reports run_emu

# test 100 samples, verifies final output only, silence excess logs
$ TARGET=sw_emu GRAPH=lenet_mnist make graph kernels xsa application package DOUT=0 DLOG=0
$ TARGET=sw_emu GRAPH=lenet_mnist make clean_reports run_emu ITER_CNT=100
```

Truncated stdout (for float32 model)
```
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
...
Kernel Name: mm2s_0, CU Number: 0, State: Start
Kernel Name: mm2s_0, CU Number: 0, State: Running
Kernel Name: s2mm_0, CU Number: 1, State: Start
Kernel Name: s2mm_0, CU Number: 1, State: Running
...
Running Conv5x5on8ReluBCHW<28, 24, 1, 1, 6>
start = -1,end = -1,total = 0
Running Maxpool2x2FloatBCHW::filter<24, 12, 1, 6>
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
...

```

Files (for float32 model)
```
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
For graph test, ITER_CNT is an argument into graph compilation (make graph). aiesimulator is cycle accurate, `DOUT=1 (default)` ensures each kernel outputs cycle information. It takes ~5 min to compile and ~5 min to run.
```
$ TARGET=hw_emu GRAPH=lenet_mnist make graph
$ TARGET=hw_emu GRAPH=lenet_mnist make clean_reports aiesim_profile

```

Truncated stdout (for float32 model)
```
Waiting for core(s) of graph lenet_mnist to finish execution ...
Running Conv5x5on8ReluBCHW<28, 24, 1, 1, 6>
start = 3236,end = 20164,total = 16928
Running Maxpool2x2FloatBCHW::filter<24, 12, 1, 6>
start = 24570,end = 25471,total = 901
...
Running GemmReluMKKN<1, 120, 32>                                # chunk graph splits computation
Running GemmReluMKKN<1, 120, 32>
Running GemmReluMKKN<1, 120, 32>
start = 56393,end = 57281,total = 888
start = 56397,end = 57285,total = 888
start = 56401,end = 57289,total = 888
...
core(s) are done executing
[wait graph]: success
plout[0]: Cycles 10 Throughput 1000000000.000000 samples/s      # event api has some issues
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
...

```

Files (for float32 model)
```
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

Truncated stdout (for float32 model)
```
INFO::[ XTLM_IPC_AXIS_MASTER ] SIM-IPC's external process can be connected to instance :  plin0_lenet_mnist_input
INFO::[ XTLM_IPC_AXIS_SLAVE ] SIM-IPC's external process can be connected to instance :  plout0_lenet_mnist_k18gemm
...
[INFO]: Enabled Stream Switch Port Latency 
18:45:08: Creating ipc_axis_master_util for plin0_lenet_mnist_input...
18:45:08: Creating ipc_axis_slave_util for plout0_lenet_mnist_k18gemm...
...
18:45:09: Begin run...
18:45:09: Running master plin0_lenet_mnist_input
18:45:09: Running slave plout0_lenet_mnist_k18gemm
...
18:45:09: [plin0_lenet_mnist_input]: Sending 392 float32 data...
...
18:47:35: [plout1_lenet_mnist_k0conv]: read lines #0
18:47:46: [plout2_lenet_mnist_k2pool]: read lines #0
...
18:51:01: Slave plout0_lenet_mnist_k18gemm finished. Written to /home/ruien/workspace/test/reports_dir/lenet_mnist/hw_emu/aiesimulator_output/k18gemm_goldenout.txt
18:51:01: Slave plout1_lenet_mnist_k0conv finished. Written to /home/ruien/workspace/test/reports_dir/lenet_mnist/hw_emu/aiesimulator_output/k0conv_goldenout.txt
...
18:51:01: Master plin0_lenet_mnist_input finished.
WARNING::[ XTLM_IPC::006 ] plin0_lenet_mnist_input Closing Socket
WARNING::[ XTLM_IPC::006 ] plout5_lenet_mnist_k14gemm Closing Socket
...

```

Extra files (for float32 model)
```
reports_dir/
└── lenet_mnist
    └── hw_emu
        └── aiesimulator_output
            └── aiesim_filetraffic.log
```

<b>Get throughput estimation</b>
```
python throughput.py reports_dir/lenet_mnist_int8/hw_emu/aiesimulator_output/k9dequantizeLinear_goldenout_shape1x10.txt
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
For system test, ITER_CNT is an argument into host app compilation (make application). This uses same aiengine graph for computation and outputs cycle information per kernel. It takes ~15 min to compile and ~30 min to run for single sample. hw_emu is known to be very slow, do not use too much data.
```
$ TARGET=hw_emu GRAPH=lenet_mnist make graph kernels xsa application package
$ TARGET=hw_emu GRAPH=lenet_mnist make graph clean_reports run_emu
```

Truncated stdout (for float32 model)
```
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
...
INFO: TEST PASSED, RC=0

9 minutes and 36 seconds elapsed.

INFO: Embedded host run completed.
```

Files (for float32 model)
```
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
It takes ~1 hour to compile, ~8 min to flash, and runs quickly.

<b>Hardware: create hardware image, flash the SD card</b>

```
# test 1 sample, verify intermediates
$ TARGET=hw GRAPH=lenet_mnist make graph kernels xsa application package EN_TRACE=1
$ sudo dd if=build/lenet_mnist/hw/package/sd_card.img of=/dev/(DEVICE) conv=fsync status=progress

# test 100 samples, verify final output
$ TARGET=hw GRAPH=lenet_mnist make graph kernels xsa application package EN_TRACE=1 DOUT=0
$ sudo dd if=build/lenet_mnist/hw/package/sd_card.img of=/dev/(DEVICE) conv=fsync status=progress
```

Truncated stdout (for float32 model), 100 samples with final output only
```
Initializing ADF API...

Config:
xclbin: a.xclbin
iter_cnt: 100
data_dir: data/
out_dir: output/

[  321.289223] zocl-drm axi:zyxclmm_drm: zocl_create_client: created KDS client for pid(607), ret: 0
[  321.298144] zocl-drm axi:zyxclmm_drm: zocl_destroy_client: client exits pid(607)
...
[  321.817940] [drm] found kind 29(AIE_RESOURCES)
[  321.817958] [drm] found kind 8(IP_LAYOUT)
[  321.822415] [drm] found kind 9(DEBUG_IP_LAYOUT)
[  321.826428] [drm] found kind 25(AIE_METADATA)
[  321.830973] [drm] found kind 7(CONNECTIVITY)
[  321.835332] [drm] found kind 6(MEM_TOPOLOGY)
[  321.839680] [drm] Memory 0 is not reserved in device tree. Will allocate memory from CMA
[  321.860410] zocl_irq_intc ZOCL_CU_INTC.2.auto: zocl_irq_intc_add: managing IRQ 46
[  321.867950] cu_drv CU.3.auto: cu_probe: CU[0] created
...
[  321.886028] cu_drv CU.3.auto:  ffff0000046f5010 xrt_cu_intr_thread: CU[0] start
...
XAIEFAL: INFO: Resource group Avail is created.
XAIEFAL: INFO: Resource group Static is created.
XAIEFAL: INFO: Resource group Generic is created.
[  321.951985] [drm] bitstream 132ce03f-bd7d-3a1a-4e13-41b7650cdd76 locked, ref=1
[  321.952008] zocl-drm axi:zyxclmm_drm:  ffff000001bfdc10 kds_add_context: Client pid(607) add context Domain(65535) CU(0xffff) shared(true)
[  321.971758] zocl-drm axi:zyxclmm_drm:  ffff000001bfdc10 kds_del_context: Client pid(607) del context Domain(65535) CU(0xffff)
...
XRT build version: 2.14.0
Build hash: 43926231f7183688add2dccfd391b36a1f000bea
Build date: 2022-10-07 05:12:02
Git branch: 2022.2
PID: 607
UID: 0
[Wed Apr 12 02:37:29 2023 GMT]
HOST:
EXE: /run/media/mmcblk0p1/aie_xrt.elf
[XRT] WARNING: The xrt.ini flag "aie_profile_core_metrics" is deprecated  and will be removed in future release. Please use tile_based_aie_metrics under "AIE_profile_settings" section.
[XRT] WARNING: Only 3 out of 4 metrics were available for aie profiling due to resource constraints. AIE profiling uses performance counters which could be already used by AIE trace, ECC, etc.
Available metrics : ACTIVE_CORE GROUP_CORE_STALL_CORE INSTR_VECTOR_CORE
Unavailable metrics : GROUP_CORE_PROGRAM_FLOW
[XRT] WARNING: The xrt.ini flag "aie_profile_memory_metrics" is deprecated  and will be removed in future release. Please use tile_based_aie_memory_metrics under "AIE_profile_settings" section.
[XRT] WARNING: The xrt.ini flag "aie_profile_interface_metrics" is deprecated  and will be removed in future release. Please use tile_based_interface_tile_metrics under "AIE_profile_settings" section.
Input0 memory virtual addr 0x0xffff69d23000
[  321.983072] [drm] bitstream 132ce03f-bd7d-3a1a-4e13-41b7650cdd76 unlocked, ref=0
[  322.169737] [drm] bitstream 132ce03f-bd7d-3a1a-4e13-41b7650cdd76 locked, ref=1
[  322.177363] zocl-drm axi:zyxclmm_drm:  ffff000001bfdc10 kds_add_context: Client pid(607) add context Domain(0) CU(0x1) shared(true)
Output0 memory virtual addr 0x0xf[  322.197006] zocl-drm axi:zyxclmm_drm:  ffff000001bfdc10 kds_add_context: Client pid(607) add context Domain(0) CU(0x0) shared(true)
fff86c28000
[init lenet_mnist]: success
[  322.215820] aie aiepart_0_50: invalid resource req(24,0),mod:2,rsc:0,expect=1 not avail.
[AIE WARNING] _XAie_LinuxIO_RequestRsc():1170: Failed to request[  322.229089] zocl-drm axi:zyxclmm_drm:  ffff000001bfdc10 kds_del_context: Client pid(607) del context Domain(0) CU(0x1)
 resource 0
[AIE WARNING] _XAie_R[  322.242710] zocl-drm axi:zyxclmm_drm:  ffff000001bfdc10 kds_del_context: Client pid(607) del context Domain(0) CU(0x0)
scMgr_RequestRsc():722: Unable t[  322.256208] [drm] bitstream 132ce03f-bd7d-3a1a-4e13-41b7650cdd76 unlocked, ref=0
[  322.256364] FAT-fs (mmcblk0p1): error, fat_free_clusters: deleting FAT entry beyond EOF

XAIEFAL: WARN: perfcount _reser[  322.277114] FAT-fs (mmcblk0p1): Filesystem has been set read-only\
ve (24,0) Expect Mod= 2 resource not available.
[XRT] ERROR: ERROR: event::start_profiling: Failed to request performance counter resources.:[  322.296122] zocl-drm axi:zyxclmm_drm: zocl_destroy_client: client exits pid(607)
 Resource temporarily unavailable
[run graph]: success
[wait graph]: success
plout[0]: ERROR: Invalid handle. Only two performance counter in a AIE-PL interface tile. Event profile is not supported for x86sim.
[end lenet_mnist]: success
Waiting for dma hls to complete...
mm2s completed with status (4)
s2mm completed with status (4)
Closed runtime handlers and kernel handlers...
[  322.393929] zocl-drm axi:zyxclmm_drm: zocl_destroy_client: client exits pid(607)
Checking directories data and output
Checking 1/1: k18gemm_goldenout_host.txt
TEST (tolerance): OK! (rtol=1e-03, atol=1e-05)
INFO: TEST PASSED, RC=0

0 minutes and 3 seconds elapsed.

Wed Apr 12 02:37:31 UTC 2023

INFO: Embedded host run completed.
```

Files (for float32 model), 100 samples with final output only
```
outputs/
├── aie_profile_edge_heat_map_conflicts_input_bandwidths_chan0.csv
├── output
│   └── k18gemm_goldenout_host.txt
├── summary.csv
└── xrt.run_summary
```

<b>Analyze system compilation after compile</b>
```
vitis_analyzer build/lenet_mnist/hw/vck190_lenet_mnist_aie.xsa.link_summary
```
<b>Analyze system profile after run</b>
```
copy file outputs from sd card to your local system
vitis_analyzer xrt.run_summary
```
<b>Clean hardware emulation build</b>
```
TARGET=hw_emu GRAPH=lenet_mnist make clean
```

## float32 vs int8 model tradeoff (WIP)
A comparison of the performance accuracy tradeoff is as follows.
```
TBC
```
aiesimulator cycle estimation per kernel
```
# float32
Running Conv5x5on8ReluBCHW<28, 24, 1, 1, 6>
start = 3221,end = 20149,total = 16928
Running Maxpool2x2FloatBCHW::filter<24,24,12,12,1,6>
start = 24557,end = 25458,total = 901
Running Conv5x5on8ReluBCHW<12, 8, 1, 6, 16>
start = 26639,end = 52715,total = 26076
Running Maxpool2x2FloatBCHW::filter<8,8,4,4,1,16>
start = 54090,end = 54381,total = 291
Running GemmReluMKKN<1,256,16>
Running GemmReluMKKN<1,256,16>
Running GemmReluMKKN<1,256,16>
Running GemmReluMKKN<1,256,16>
Running GemmReluMKKN<1,256,16>
Running GemmReluMKKN<1,256,16>
Running GemmReluMKKN<1,256,16>
Running GemmReluMKKN<1,256,16>
start = 54809,end = 55712,total = 903
start = 54813,end = 55716,total = 903
start = 54813,end = 55716,total = 903
start = 54817,end = 55720,total = 903
start = 54821,end = 55724,total = 903
start = 54825,end = 55728,total = 903
start = 54829,end = 55732,total = 903
start = 54833,end = 55736,total = 903
Running ConcatVector<8,1,16,120>::filter8
start = 55936,end = 56062,total = 126
Running GemmReluMKKN<1,120,32>
Running GemmReluMKKN<1,120,32>
Running GemmReluMKKN<1,120,32>
start = 56382,end = 57270,total = 888
start = 56386,end = 57274,total = 888
start = 56390,end = 57278,total = 888
Running ConcatVector<3,1,32,84>::filter3
start = 57426,end = 57499,total = 73
Running GemmReluMKKN<1,84,48>
start = 57771,end = 58744,total = 973
Running ConcatScalar<1,1,48,10>::filter1
start = 58888,end = 59766,total = 878

# int8
Running QuantizeLinearVector<28,28,32>
start = 3035,end = 5492,total = 2457
Running QLinearConvVector<28,32,24,32,1,1,6,5>
start = 5884,end = 10201,total = 4317
Running Maxpool2x2Int8BCHW::filter<24,32,12,16,1,6>
start = 11729,end = 12053,total = 324
Running QLinearConvVector<12,16,8,16,1,6,16,5>
start = 12521,end = 19227,total = 6706
Running MaxpoolScalarBCHW::filter<8,16,4,4,1,16>
start = 19960,end = 23554,total = 3594
Running QgemmVector<1,256,128>
start = 23790,end = 25977,total = 2187
Running QgemmVector<1,128,96>
start = 26173,end = 27057,total = 884
Running QgemmVector<1,96,16>
start = 27257,end = 27396,total = 139
Running DequantizeLinearScalar<16,10>
start = 27560,end = 27642,total = 82
...
```