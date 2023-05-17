# onnx2versal

This repo holds AIE kernels/graphs and generator scripts to create system level design given a ONNX model and some data. Verify, profile and run your ONNX models on Versal machines! 

See example at [Lenet Example](lenet_example.md) <br />
See documentation at https://rehohoho.github.io/onnx2versal/: actual kernel and graph details relevant only if you are developing on this repo.

## Requirements
* Vitis Software Platform 2022.2
* xilinx-versal-common image
* X86 XRT (only for software system emulation)
* Python 3

## Setup
Setup following lines in `./sample_env_setup.sh`. Run `source sample_env_setup.sh` to setup environment in your shell.
```
export PLATFORM_REPO_PATHS=/tools/Xilinx/Vitis/2022.2/base_platforms
export XILINX_VITIS=/tools/Xilinx/Vitis/2022.2
export COMMON_IMAGE_VERSAL=/tools/xilinx-versal-common-v2022.2
export XILINX_X86_XRT=/opt/xilinx/xrt
export PYTHON3_LOCATION=/usr/bin
```

## Usage

**Inputs**
```
models/my_model.onnx
some data to run through ONNX model
```

**Runnables**
```
python/generate.py
```

**Outputs**
```
design/aie_src/graph_[net_name].cpp               # main .cpp to be compiled by aiecompiler
design/host_app_src/[net_name]_aie_app.cpp        # host .cpp for PS
design/system_configs/[net_name]_output_inter.cfg # config files for v++ system linking
design/system_configs/[net_name].cfg
design/traffic_gen/xtg_[net_name]_output_inter.py # external traffic generator files for tests
design/traffic_gen/xtg_[net_name].py
```

**Simulation, Verification, Profiling, Hardware Build** <br />
`make help` for full details. Only key (and probably insufficient) examples here.
```
## Functional check: runs graph in x86simulator
$ TARGET=sw_emu GRAPH=my_model [EXTIO=1] [DOUT=0] make graph clean_reports aiesim

## Functional check: runs system software emulation (x86 graph, sysC kernels)
$ TARGET=sw_emu GRAPH=my_model [DOUT=0] make graph kernels xsa application package clean_reports run_emu


## Functional/performance check: runs graph in aiesimulator
$ TARGET=hw_emu GRAPH=my_model [EXTIO=1] [DOUT=0] make graph clean_reports aiesim_profile

## Functional/performance check: runs system in hardware emulation with SysC QEMU (sysC graph, kernels NoC, DDR)
$ TARGET=hw_emu GRAPH=my_model [DOUT=0] make graph kernels xsa application package run_emu


## Hardware: create hardware image, flash the SD card
$ TARGET=hw GRAPH=my_model [DOUT=0] make graph kernels xsa application package
$ sudo dd if=build/lenet/sw_emu/package/sd_card.img of=/dev/ conv=fsync status=progress
```

## Directory structure
```
design/
├── aie_src
│   ├── *.cc            # kernel
│   ├── *.h
│   ├── graph*.cpp      # unit test
│   └── graph*.h        # graph
├── directives          # directives for linking
├── exec_scripts        # bash scripts to be packaged
├── host_app_src
│   └── *_aie_app.cpp   # host script for PS, used in system run
├── pl_src              # data moving kernels
├── profiling_configs   # xrt.inis for hardware profiling
├── system_configs      # .cfg for v++ linking
│   ├── *.cfg
│   └──  *_output_inter.cfg
├── trafficgen          # traffic gen python scripts for graph runs
│   ├── xtg_*.py
│   └──  xtg_*_output_inter.py
└── vivado_metrics_scripts
```

## Let me dev

### Important documentation
* [Vitis Tutorials - AI Engine Development](https://github.com/Xilinx/Vitis-Tutorials/tree/2022.1/AI_Engine_Development)
* [AI Engine Kernel and Graph Programming Guide (UG1079)](https://docs.xilinx.com/r/en-US/ug1079-ai-engine-kernel-coding/)
* [AI Engine Documentation - AIE API or Intrinsic guide](https://www.xilinx.com/htmldocs/aiengine_intrinsics_start.html)
* [AI Engine Tools and Flows User Guide (UG1076)](https://docs.xilinx.com/r/en-US/ug1076-ai-engine-environment/)
* [Vitis HLS](https://docs.xilinx.com/r/en-US/ug1399-vitis-hls)
* [XRT Documentation - Host application programming docs](https://xilinx.github.io/XRT/master/html/index.html)
* [XRT AIE API - xrt_aie.h](https://github.com/Xilinx/XRT/blob/master/src/runtime_src/core/include/experimental/xrt_aie.h)

### Write a graph
See reference generated graph from example. See documentation for any dimension restrictions for kernels/graphs.

### Write a op
```
design/aie_src
├── my_op.cc
├── my_op.h
├── graph_my_op.h
└── graph_my_op.cpp
```

### Add data
```
data/
├── my_op_in.txt
└── my_op_golden.txt
```

### Test the op
```
# X86 GRAPH
TARGET=sw_emu GRAPH=my_op make clean_reports graph aiesim

# SYSC GRAPH
TARGET=hw_emu GRAPH=my_op make clean_reports graph aiesim
```
