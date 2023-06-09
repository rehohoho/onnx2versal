# © Copyright 2022 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

%.PHONY: help
help::
	@echo  ""
	@echo  " Makefile Usage:"
	@echo  ""
	@echo  " OPTIONS:"
	@echo  " Use the make recipes with required values for options mentioned below-"
	@echo  "    TARGET      sw_emu(default)|hw_emu|hw, build target"
	@echo  "    GRAPH       lenet_mnist (default),     target graph as per design/aie_src/graph_[].cpp"
	@echo  "    EXTIO       0 (default) | 1,           traffic gen usage, graph runs only, redundant for system due to host script"
	@echo  "    DLOG        1 (default) | 0,           if enable verbose logging, kernel names and timing"
	@echo  "    DOUT        1 (default) | 0,           if enable output intermediates, max 6-7 outputs, AIE has <= 8 cascade channels"
	@echo  "    ITER_CNT    1 (default),               number of run iterations, independent in x86sim, aiesim, emu"
	@echo  "    EN_TRACE    0 (default) | 1,           enable profiling .ini (hw)"
	@echo  ""
	@echo  " Recipes: "
	@echo  "  graph           compiles using aiecompiler on design/aie_src/graph_(GRAPH).cpp"
	@echo  "                  outputs *.a *.log Work/ in build/(GRAPH)/(TARGET)"
	@echo  ""
	@echo  "  aiesim          runs x86simulator or aiesimulator on compiled (GRAPH)"
	@echo  "                  outputs *.txt *.log in reports_dir/(GRAPH)/(TARGET)"
	@echo  ""
	@echo  "  aiesim_profile  runs x86simulator or aiesimulator on compiled (GRAPH) with --profile"
	@echo  "                  outputs *.txt *.log in reports_dir/(GRAPH)/(TARGET)"
	@echo  ""
	@echo  "  kernels         compiles using v++ on design/pl_src/*.cpp"
	@echo  "                  outputs *.xo *.log *.compile_summary in build/(GRAPH)/(TARGET)"
	@echo  ""
	@echo  "  xsa             links using v++ -l on compiled (GRAPH) and kernels, configs on design/directives design/system_configs"
	@echo  "                  outputs *.xsa *.log *.link_summary in build/(GRAPH)/(TARGET)"
	@echo  ""
	@echo  "  application     compiles using (CXX) on design/host_app_src/(GRAPH)_aie_app.cpp"
	@echo  "                  outputs *.o *.elf in build/(GRAPH)/(TARGET)"
	@echo  ""
	@echo  "  package         packages using v++ on *.xsa *.a design/exec_scripts/"
	@echo  "                  outputs *.xclbin or package/ in build/(GRAPH)/(TARGET)"
	@echo  ""
	@echo  "  run_emu         sets environments variables and runs packaged emulation"
	@echo  ""
	@echo  "  clean           cleans build/$(GRAPH)/$(TARGET)"
	@echo  ""
	@echo  "  clean_reports   cleans reports_dir/$(GRAPH)/$(TARGET)"
	@echo  ""
	@echo  ""
	@echo  " Examples: [] are optional"
	@echo  ""
	@echo  "  Functional check: runs graph in x86simulator"
	@echo  "  $$ TARGET=sw_emu GRAPH=lenet_mnist [EXTIO=1] [DOUT=0] make graph clean_reports aiesim"
	@echo  ""
	@echo  "  Functional check: runs system software emulation (x86 graph, sysC kernels)"
	@echo  "  $$ TARGET=sw_emu GRAPH=lenet_mnist [DOUT=0] make graph kernels xsa application package clean_reports run_emu"
	@echo  ""
	@echo  "  Clean software build and reports."
	@echo  "  $$ TARGET=sw_emu make clean clean_reports"
	@echo  ""
	@echo  ""
	@echo  "  Functional/performance check: runs graph in aiesimulator"
	@echo  "  $$ TARGET=hw_emu GRAPH=lenet_mnist [EXTIO=1] [DOUT=0] make graph clean_reports aiesim_profile"
	@echo  ""
	@echo  "  Functional/performance check: runs system in hardware emulation with QEMU (sysC graph, kernels NoC, DDR)"
	@echo  "  $$ TARGET=hw_emu GRAPH=lenet_mnist [DOUT=0] make graph kernels xsa application package run_emu"
	@echo  ""
	@echo  "  Clean hardware emulation build and reports."
	@echo  "  $$ TARGET=hw_emu make clean clean_reports"
	@echo  ""
	@echo  ""
	@echo  "  Hardware: create hardware image, flash the SD card"
	@echo  "  $$ TARGET=hw GRAPH=lenet_mnist [DOUT=0] make graph kernels xsa application package"
	@echo  "  $$ sudo dd if=build/lenet_mnist/hw/package/sd_card.img of=/dev/(DEVICE) conv=fsync status=progress"
	@echo  ""

# Print all options passed to Makefile
print-%  : ; @echo $* = $($*)

# =========================================================
# TARGET can be set as:
#   hw_emu: Hardware Emulation
#   hw    : Hardware Run
# =========================================================
TARGET ?= sw_emu
GRAPH ?= lenet_mnist
EXTIO ?= 0
DLOG ?= 1
DOUT ?= 1
ITER_CNT ?= 1
EN_TRACE ?= 0

# =========================================================
# PL Frequency in Mhz:
#   312.5Mhz(default)
# =========================================================
PL_FREQ := 312.5
HZ_UNIT := 1000000
VPP_CLOCK_FREQ := $(shell printf "%.0f" `echo "${PL_FREQ} * $(HZ_UNIT)" | bc`)
#VPP_CLOCK_FREQ := $(PL_FREQ)000000

# =========================================================
# Source directories
# =========================================================
RELATIVE_PROJECT_DIR := ./
# Absolute net directory
PROJECT_REPO := $(shell readlink -f $(RELATIVE_PROJECT_DIR))

DESIGN_REPO  := $(PROJECT_REPO)/design
DATA_REPO := $(PROJECT_REPO)/data
AIE_SRC_REPO := $(DESIGN_REPO)/aie_src
PL_SRC_REPO  := $(DESIGN_REPO)/pl_src
HOST_APP_SRC_REPO := $(DESIGN_REPO)/host_app_src

DIRECTIVES_REPO        := $(DESIGN_REPO)/directives
SYSTEM_CONFIGS_REPO    := $(DESIGN_REPO)/system_configs
PROFILING_CONFIGS_REPO := $(DESIGN_REPO)/profiling_configs
EXEC_SCRIPTS_REPO      := $(DESIGN_REPO)/exec_scripts
VIVADO_METRICS_SCRIPTS_REPO := $(DESIGN_REPO)/vivado_metrics_scripts

BUILD_TARGET_DIR  := $(PROJECT_REPO)/build/$(GRAPH)/$(TARGET)

REPORTS_REPO := $(PROJECT_REPO)/reports_dir
BLD_REPORTS_DIR := $(REPORTS_REPO)/$(GRAPH)/$(TARGET)

XPE_REPO         := $(PROJECT_REPO)/xpe_dir

WORK_DIR := Work

# =========================================================
# Step 1. Kernel XO File Generation
# ========================================================
# This step compiles the HLS C PL kernels. 
# Outputs: in build/[hw_emu | hw]/ directory
# 	dma_hls.[hw_emu | hw].xo  
#	dma_hls.[hw_emu | hw].xo.compile_summary  
#	v++_dma_hls.[hw_emu | hw].log
#	_x
KERNEL1 := float32_s2mm
KERNEL2 := float32_mm2s
KERNEL3 := int8_s2mm
KERNEL4 := int8_mm2s
KERNEL1_XO  := $(KERNEL1).$(TARGET)
KERNEL2_XO  := $(KERNEL2).$(TARGET)
KERNEL3_XO  := $(KERNEL3).$(TARGET)
KERNEL4_XO  := $(KERNEL4).$(TARGET)
KERNEL1_SRC := $(PL_SRC_REPO)/float32_s2mm.cpp
KERNEL2_SRC := $(PL_SRC_REPO)/float32_mm2s.cpp
KERNEL3_SRC := $(PL_SRC_REPO)/int8_s2mm.cpp
KERNEL4_SRC := $(PL_SRC_REPO)/int8_mm2s.cpp

TRAFFICGEN_WIDTH := 32

VPP_FLAGS := --platform $(PLATFORM) \
					   --save-temps \
					   --temp_dir $(BUILD_TARGET_DIR)/_x \
					   --verbose \
					   -g

ifeq ($(EXTIO), 1)
	KERNEL_XOS := $(XILINX_VITIS)/data/emulation/XO/sim_ipc_axis_master_$(TRAFFICGEN_WIDTH).xo \
								$(XILINX_VITIS)/data/emulation/XO/sim_ipc_axis_slave_$(TRAFFICGEN_WIDTH).xo
else
	KERNEL_XOS := $(BUILD_TARGET_DIR)/$(KERNEL1_XO).xo \
							  $(BUILD_TARGET_DIR)/$(KERNEL2_XO).xo \
							  $(BUILD_TARGET_DIR)/$(KERNEL3_XO).xo \
							  $(BUILD_TARGET_DIR)/$(KERNEL4_XO).xo
endif

kernels: $(KERNEL_XOS)

$(BUILD_TARGET_DIR)/$(KERNEL1_XO).xo:
	mkdir -p $(BUILD_TARGET_DIR); cd $(BUILD_TARGET_DIR); \
	v++ --target $(TARGET) $(VPP_FLAGS) -c --hls.clock $(VPP_CLOCK_FREQ):$(KERNEL1) -k $(KERNEL1) $(KERNEL1_SRC) -o $@

$(BUILD_TARGET_DIR)/$(KERNEL2_XO).xo:
	mkdir -p $(BUILD_TARGET_DIR); cd $(BUILD_TARGET_DIR); \
	v++ --target $(TARGET) $(VPP_FLAGS) -c --hls.clock $(VPP_CLOCK_FREQ):$(KERNEL2) -k $(KERNEL2) $(KERNEL2_SRC) -o $@

$(BUILD_TARGET_DIR)/$(KERNEL3_XO).xo:
	mkdir -p $(BUILD_TARGET_DIR); cd $(BUILD_TARGET_DIR); \
	v++ --target $(TARGET) $(VPP_FLAGS) -c --hls.clock $(VPP_CLOCK_FREQ):$(KERNEL3) -k $(KERNEL3) $(KERNEL3_SRC) -o $@

$(BUILD_TARGET_DIR)/$(KERNEL4_XO).xo:
	mkdir -p $(BUILD_TARGET_DIR); cd $(BUILD_TARGET_DIR); \
	v++ --target $(TARGET) $(VPP_FLAGS) -c --hls.clock $(VPP_CLOCK_FREQ):$(KERNEL4) -k $(KERNEL4) $(KERNEL4_SRC) -o $@

# =========================================================
# Step 2. AI Engine SDF Graph File and Work/ Directory 
#         (containing the Graph Executable) Generation
# ========================================================
# This step creates an SDF Graph and the Work/ directory.
# The Work/ directory contains the graph executable 
# (net.o) which is used in the make xsa step.  
# The aiecompiler is invoked with the -target=hw. 
# Outputs: in build/ directory
#	libsdf.a 
#	NOC_Power.xpe
#	Work/ 
#	xnwOut/
LIBADF_A 			:= $(BUILD_TARGET_DIR)/libadf.a
GRAPH_SRC_CPP := $(AIE_SRC_REPO)/graph_$(GRAPH).cpp

# heapsize + stacksize (1024 default) + syncsize = 32768 bytes max
# heapsize of 16384 bytes allows maximum of 4096 float params per AIE
AIE_FLAGS := -include=$(AIE_SRC_REPO) \
						 --verbose \
						 --Xpreproc="-DITER_CNT=$(ITER_CNT)" \
						 --Xchess="main:backend.mist2.maxfoldk=256" \
						 --platform=$(PLATFORM) \
						 --log-level=5 \
						 --pl-freq=500 \
						 --dataflow \
						 --stacksize=1024 \
						 --heapsize=1024 \
						 --workdir=$(WORK_DIR) \
						 --evaluate-fifo-depth
ifeq ($(TARGET), sw_emu)
	AIE_FLAGS += --target=x86sim --Xpreproc=-O0
endif
ifeq ($(EXTIO), 1)
	AIE_FLAGS += --Xpreproc=-DEXTERNAL_IO
endif
ifeq ($(DLOG), 1)
	AIE_FLAGS += --Xpreproc=-D__LOG_VERBOSE__
endif
ifeq ($(DOUT), 1)
	AIE_FLAGS += --Xpreproc=-D__OUTPUT_INTER__
endif
#AIE_FLAGS += --test-iterations=100 
#AIE_FLAGS += --Xmapper=BufferOptLevel9
#AIE_FLAGS += --Xrouter=DMAFIFOsInFreeBankOnly

graph: $(LIBADF_A)

$(LIBADF_A): $(AIE_SRC_REPO)/*
	mkdir -p $(BUILD_TARGET_DIR); \
	cd $(BUILD_TARGET_DIR); \
	aiecompiler $(AIE_FLAGS) $(GRAPH_SRC_CPP) 2>&1 | tee -a aiecompiler.log

# =========================================================
# Step 2b. AI Engine Simulation
# ========================================================
# Aiesimulator flags
#-i - Alias of --input-dir=<dir> option
#-o - Alias of --output-dir=<dir> option
AIE_SIM_FLAGS := --pkg-dir=$(BUILD_TARGET_DIR)/$(WORK_DIR)/ \
								 -i=$(DATA_REPO)
X86SIM_REPORT_DIR := $(BLD_REPORTS_DIR)/x86simulator_output
AIESIM_REPORT_DIR := $(BLD_REPORTS_DIR)/aiesimulator_output

ifeq ($(DOUT), 1)
TRAFFIC_GEN_PY := $(DESIGN_REPO)/trafficgen/xtg_$(GRAPH)_output_inter.py
else
TRAFFIC_GEN_PY := $(DESIGN_REPO)/trafficgen/xtg_$(GRAPH).py
endif

aiesim: graph
ifeq ($(EXTIO), 0)
ifeq ($(TARGET), sw_emu)
	mkdir -p $(X86SIM_REPORT_DIR); \
	cd $(BUILD_TARGET_DIR); \
	x86simulator $(AIE_SIM_FLAGS) -o=$(X86SIM_REPORT_DIR); \
	python3 $(PROJECT_REPO)/check.py -f1=$(X86SIM_REPORT_DIR) -f2=$(DATA_REPO) 2>&1 | tee -a $(X86SIM_REPORT_DIR)/x86sim_check.log
else
	mkdir -p $(AIESIM_REPORT_DIR); \
	cd $(BUILD_TARGET_DIR); \
	aiesimulator $(AIE_SIM_FLAGS) -o $(AIESIM_REPORT_DIR) 2>&1 | tee -a $(AIESIM_REPORT_DIR)/aiesim.log; \
	python3 $(PROJECT_REPO)/check.py -f1=$(AIESIM_REPORT_DIR) -f2=$(DATA_REPO) 2>&1 | tee -a $(AIESIM_REPORT_DIR)/aiesim_check.log
endif
else  # Use External Traffic Generator
ifeq ($(TARGET), sw_emu)
	mkdir -p $(X86SIM_REPORT_DIR); \
	cd $(BUILD_TARGET_DIR); \
	x86simulator $(AIE_SIM_FLAGS) -o=$(X86SIM_REPORT_DIR) & python3 $(TRAFFIC_GEN_PY) --input_dir $(DATA_REPO) --output_dir $(X86SIM_REPORT_DIR) 2>&1 | tee -a $(X86SIM_REPORT_DIR)/x86sim_filetraffic.log; \
	python3 $(PROJECT_REPO)/check.py -f1=$(X86SIM_REPORT_DIR) -f2=$(DATA_REPO) 2>&1 | tee -a $(X86SIM_REPORT_DIR)/x86sim_check.log
else
	mkdir -p $(AIESIM_REPORT_DIR); \
	cd $(BUILD_TARGET_DIR); \
	aiesimulator $(AIE_SIM_FLAGS) -o $(AIESIM_REPORT_DIR) 2>&1 | tee -a $(AIESIM_REPORT_DIR)/aiesim.log & python3 $(TRAFFIC_GEN_PY) --input_dir $(DATA_REPO) --output_dir $(AIESIM_REPORT_DIR) 2>&1 | tee -a $(AIESIM_REPORT_DIR)/aiesim_filetraffic.log; \
	python3 $(PROJECT_REPO)/check.py -f1=$(AIESIM_REPORT_DIR) -f2=$(DATA_REPO) 2>&1 | tee -a $(AIESIM_REPORT_DIR)/aiesim_check.log
endif
endif

aiesim_profile: graph
ifeq ($(EXTIO), 0)
ifeq ($(TARGET), sw_emu)
	@echo "sw_emu does not support profiling, requires hw | hw_emu"
else
	mkdir -p $(AIESIM_REPORT_DIR); \
	cd $(BUILD_TARGET_DIR); \
	aiesimulator $(AIE_SIM_FLAGS) --profile -o $(AIESIM_REPORT_DIR) 2>&1 | tee -a $(AIESIM_REPORT_DIR)/aiesim.log; \
	python3 $(PROJECT_REPO)/check.py -f1=$(AIESIM_REPORT_DIR) -f2=$(DATA_REPO) 2>&1 | tee -a $(AIESIM_REPORT_DIR)/x86sim_check.log
endif
else  # Use External Traffic Generator
ifeq ($(TARGET), sw_emu)
	@echo "sw_emu does not support profiling, requires hw | hw_emu"
else
	mkdir -p $(AIESIM_REPORT_DIR); \
	cd $(BUILD_TARGET_DIR); \
	aiesimulator $(AIE_SIM_FLAGS) --profile -o $(AIESIM_REPORT_DIR) 2>&1 | tee -a $(AIESIM_REPORT_DIR)/aiesim.log & python3 $(TRAFFIC_GEN_PY) --input_dir $(DATA_REPO) --output_dir $(AIESIM_REPORT_DIR) 2>&1 | tee -a $(AIESIM_REPORT_DIR)/aiesim_filetraffic.log; \
	python3 $(PROJECT_REPO)/check.py -f1=$(AIESIM_REPORT_DIR) -f2=$(DATA_REPO) 2>&1 | tee -a $(AIESIM_REPORT_DIR)/x86sim_check.log
endif
endif


BLD_XPE_DIR      := $(XPE_REPO)/$(GRAPH)
BLD_TGT_VCD_FILE := $(BUILD_TARGET_DIR)/$(GRAPH).vcd
XPE_FILE         := $(BLD_XPE_DIR)/$(GRAPH).xpe

vcd: graph $(BLD_TGT_VCD_FILE)

# xpe file generation...
$(XPE_FILE): $(BLD_TGT_VCD_FILE)
	cd $(BUILD_TARGET_DIR); \
	vcdanalyze --vcd $(GRAPH).vcd --xpe
	rm -rf $(BLD_XPE_DIR)
	mkdir -p $(BLD_XPE_DIR)
	cp -rf $(BUILD_TARGET_DIR)/aiesim_xpe/*.xpe $(XPE_FILE)

# vcd file generation...
$(BLD_TGT_VCD_FILE):
	cd $(BUILD_TARGET_DIR); \
	aiesimulator $(AIE_SIM_FLAGS) --profile --dump-vcd $(GRAPH) 2>&1 | tee -a vcd.log; \
	vcdanalyze --vcd $(GRAPH).vcd

# =========================================================
# Step 3. XSA File Generation
# ========================================================
# This step links the graph executable (tx_chain.o) and 
# the kernels into a xsa file. 
# Outputs: in build/[hw_emu | hw]/ directory
XSA := vck190_$(GRAPH)_aie.xsa

VPP_LINK_FLAGS := --clock.defaultTolerance 0.001 \
                  --advanced.param compiler.userPostSysLinkOverlayTcl=$(DIRECTIVES_REPO)/noc_qos.tcl \
                  --vivado.prop run.synth_1.STEPS.SYNTH_DESIGN.ARGS.CONTROL_SET_OPT_THRESHOLD=16

ifeq ($(DOUT), 1)
	VPP_LINK_FLAGS += --config $(SYSTEM_CONFIGS_REPO)/$(GRAPH)_output_inter.cfg
else
	VPP_LINK_FLAGS += --clock.freqHz $(VPP_CLOCK_FREQ):$(KERNEL1)_0 \
										--clock.freqHz $(VPP_CLOCK_FREQ):$(KERNEL2)_0 \
										--clock.freqHz $(VPP_CLOCK_FREQ):$(KERNEL3)_0 \
										--clock.freqHz $(VPP_CLOCK_FREQ):$(KERNEL4)_0 \
										--config $(SYSTEM_CONFIGS_REPO)/$(GRAPH).cfg
endif

ifeq ($(EN_TRACE),1)
   ifeq ($(TARGET),hw)
      VPP_LINK_FLAGS += --profile.data $(KERNEL1):all:all \
                        --profile.data $(KERNEL2):all:all \
                        --profile.data $(KERNEL3):all:all \
                        --profile.data $(KERNEL4):all:all \
                        --profile.trace_memory DDR
   endif
endif

## Enabling Multiple Strategies For Closing Timing...
#VPP_LINK_FLAGS += --vivado.impl.strategies "ALL"
#VPP_LINK_FLAGS += --vivado.impl.lsf '{bsub -R "select[type=X86_64] rusage[mem=65536]" -N -q medium}'
#VPP_LINK_FLAGS += --advanced.param "compiler.enableMultiStrategies=1"
#VPP_LINK_FLAGS += --advanced.param "compiler.multiStrategiesWaitOnAllRuns=1"
#VPP_LINK_FLAGS += --vivado.synth.jobs 32
#VPP_LINK_FLAGS += --vivado.impl.jobs 20

xsa: kernels graph $(BUILD_TARGET_DIR)/$(XSA)
ifeq ($(EXTIO), 1)
	@echo "EXTIO not supported for system tests. Host code is traffic generator."
else
$(BUILD_TARGET_DIR)/$(XSA):$(KERNEL_XOS) $(LIBADF_A) $(SYSTEM_CONFIGS_REPO)/*
	cd $(BUILD_TARGET_DIR);	\
	v++ -l $(VPP_FLAGS) $(VPP_LINK_FLAGS) -t $(TARGET) -o $@ \
		    $(KERNEL_XOS) $(LIBADF_A)
endif

# =========================================================
# Step 4. A72 Application Executable File Generation
# ========================================================
# This step compiles the A72 application. This step is the  
# same for TARGET=[hw_emu | hw]. Compile the PS code.
# Outputs: in build/ directory
# For finite run...
# 	aie_control.o
#	  net_app.o 
# 	net_xrt.elf
APP_OBJ_NAME 		:= $(GRAPH)_aie
APP_ELF 				:= aie_xrt.elf
APP_SRC_CPP 	  := $(HOST_APP_SRC_REPO)/$(GRAPH)_aie_app.cpp
AIE_CONTROL_CPP := $(BUILD_TARGET_DIR)/$(WORK_DIR)/ps/c_rts/aie_control_xrt.cpp

GCC_FLAGS := -O \
             -c \
             -std=c++17 \
             -D__linux__ \
             -D__PS_ENABLE_AIE__ \
             -DXAIE_DEBUG

GCC_INC_FLAGS := -I$(XILINX_VITIS)/aietools/include/ \
                 -I$(AIE_SRC_REPO) \
                 -I$(HOST_APP_SRC_REPO)

GCC_INC_LIB := -ladf_api_xrt \
               -lxrt_coreutil 

ifeq ($(DOUT), 1)
GCC_FLAGS += -D__OUTPUT_INTER__
endif

ifeq ($(TARGET), sw_emu)
CXX := g++
GCC_FLAGS += -D__IS_SW_EMU__
GCC_INC_FLAGS += -I${XILINX_X86_XRT}/include
GCC_INC_LIB += -L${XILINX_X86_XRT}/lib \
								-L$(XILINX_VITIS)/aietools/lib/lnx64.o \
								-pthread \
								-lxrt_core
else
GCC_INC_FLAGS += -I$(SDKTARGETSYSROOT)/usr/include/xrt \
								 -I$(SDKTARGETSYSROOT)/usr/include \
								 -I$(SDKTARGETSYSROOT)/usr/lib

GCC_INC_LIB += -L$(SDKTARGETSYSROOT)/usr/lib \
							 -L$(XILINX_VITIS)/aietools/lib/aarch64.o \
							 -L$(XILINX_VITIS)/aietools/lib/lnx64.o
endif

application: graph $(BUILD_TARGET_DIR)/$(APP_ELF)
ifeq ($(EXTIO), 1)
	@echo "EXTIO not supported for system tests. Host code is traffic generator."
else
$(BUILD_TARGET_DIR)/$(APP_ELF): $(HOST_APP_SRC)/* $(LIBADF_A)
	@rm -rf $(BUILD_TARGET_DIR)/app_control.o $(BUILD_TARGET_DIR)/$(APP_OBJ_NAME)_app.o $(BUILD_TARGET_DIR)/$(APP_ELF)
	$(CXX) $(GCC_FLAGS) $(GCC_INC_FLAGS) $(AIE_CONTROL_CPP) -o $(BUILD_TARGET_DIR)/app_control.o
	$(CXX) $(GCC_FLAGS) $(GCC_INC_FLAGS) $(APP_SRC_CPP) -o $(BUILD_TARGET_DIR)/$(APP_OBJ_NAME)_app.o $(GCC_INC_LIB)
	$(CXX) $(BUILD_TARGET_DIR)/app_control.o $(BUILD_TARGET_DIR)/$(APP_OBJ_NAME)_app.o $(GCC_INC_LIB) -o $(BUILD_TARGET_DIR)/$(APP_ELF)
endif

# =========================================================
# Step 5. Package Generation  
# ========================================================
# This step generates the package folder which contains the 
# ./launch_hw_emu.sh script to launch hardware emulation 
# if TARGET=hw_emu and the sd_card.img file.  
# Outputs: in build/[hw_emu | hw]/ directory 
# 	a.xclbin
# 	package/ directory
# 	v++.package_summary
# 	v++_package.log 
EMBEDDED_PACKAGE_OUT := $(BUILD_TARGET_DIR)/package
EMBEDDED_EXEC_HWEMU_SCRIPT := run_hw_emu.sh
EMBEDDED_EXEC_HW_SCRIPT := run_hw.sh

PKG_FLAGS := -t $(TARGET) \
             --save-temps \
             --temp_dir $(BUILD_TARGET_DIR)/_x \
             -f $(PLATFORM) \
						 --package.defer_aie_run \
						 --package.enable_aie_debug \
						 $(BUILD_TARGET_DIR)/$(XSA) $(LIBADF_A)

ifeq ($(TARGET), sw_emu)
PKG_FLAGS += -o $(BUILD_TARGET_DIR)/a.xclbin \
					   --package.emu_ps x86
else
PKG_FLAGS += --package.rootfs $(COMMON_IMAGE_VERSAL)/rootfs.ext4 \
             --package.kernel_image $(COMMON_IMAGE_VERSAL)/Image \
             --package.boot_mode=sd \
             --package.out_dir $(EMBEDDED_PACKAGE_OUT) \
             --package.image_format=ext4 \
             --package.sd_file $(BUILD_TARGET_DIR)/$(APP_ELF) \
             --package.sd_file $(EXEC_SCRIPTS_REPO)/$(EMBEDDED_EXEC_HWEMU_SCRIPT) \
             --package.sd_file $(EXEC_SCRIPTS_REPO)/$(EMBEDDED_EXEC_HW_SCRIPT) \
						 --package.sd_file $(PROJECT_REPO)/check.py \
						 --package.sd_dir $(DATA_REPO)
endif

ifeq ($(EN_TRACE),1)
   ifeq ($(TARGET),hw)
      PKG_FLAGS += --package.sd_file $(PROFILING_CONFIGS_REPO)/xrt.ini
   endif
endif

ifdef XRT_ROOT
   PKG_FLAGS += --package.sd_dir $(XRT_ROOT)
endif

package: application xsa $(EMBEDDED_PACKAGE_OUT)
ifeq ($(EXTIO), 1)
	@echo "EXTIO not supported for system tests. Host code is traffic generator."
else
$(EMBEDDED_PACKAGE_OUT): $(PROFILING_CONFIGS_REPO)/* $(EXEC_SCRIPTS_REPO)/*
	rm -rf $(EMBEDDED_PACKAGE_OUT)
	cd $(BUILD_TARGET_DIR);	\
	v++ -p $(PKG_FLAGS)
	#cp -rf $(EMBEDDED_PACKAGE_OUT) $(EMBEDDED_PACKAGE_OUT)_backup
endif

# =========================================================
# Step 6. Run Hardware Emulation 
# ========================================================
# If the target is for HW_EMU, launch the emulator
# If the target is for HW, you'll have to follow the
# instructions in the README.md
run_emu:
ifeq ($(EXTIO), 1)
	@echo "EXTIO not supported for system tests. Host code is traffic generator."
else

ifeq ($(TARGET),hw_emu)
	cd $(EMBEDDED_PACKAGE_OUT); \
	./launch_hw_emu.sh -run-app $(EMBEDDED_EXEC_HWEMU_SCRIPT) -no-reboot | tee $(AIESIM_REPORT_DIR)/embedded_run.log
else # sw_emu
	mkdir -p $(X86SIM_REPORT_DIR); \
	cd $(BUILD_TARGET_DIR); \
	export XILINX_XRT=$(XILINX_X86_XRT); \
	export XCL_EMULATION_MODE=sw_emu; \
	export LD_LIBRARY_PATH=${XILINX_X86_XRT}/lib; \
	./$(APP_ELF) a.xclbin $(ITER_CNT) $(DATA_REPO) $(X86SIM_REPORT_DIR) 2>&1 | tee $(X86SIM_REPORT_DIR)/embedded_run.log; \
	python3 $(PROJECT_REPO)/check.py -f1=$(X86SIM_REPORT_DIR) -f2=$(DATA_REPO) 2>&1 | tee -a $(X86SIM_REPORT_DIR)/x86sim_check.log; \
	unset LD_LIBRARY_PATH
endif

endif

# =========================================================
# Step 7. Report Utilization Metrics
# =========================================================
# If the target is HW, this generates the power and resource
# utilization metrics from vivado.
report_metrics: $(VIVADO_METRICS_SCRIPTS_REPO)/report_metrics.tcl 
ifeq ($(TARGET),hw_emu)
	@echo "This build target (report-metrics) not valid when design target is hw_emu"

else
	rm -rf $(BLD_REPORTS_DIR)
	mkdir -p $(BLD_REPORTS_DIR)
	cd $(BLD_REPORTS_DIR); \
	vivado -mode batch -source $(VIVADO_METRICS_SCRIPTS_REPO)/report_metrics.tcl $(BUILD_TARGET_DIR)/_x/link/vivado/vpl/prj/prj.xpr

endif

###########################################################################

# =========================================================
# Primary Build Targets
# ==> run
# ==> all
# ==> clean
# ==> clean_reports
# ==> clean_xpe
# ========================================================

#Build the design and then run hardware emulation 
run: package run_emu

#Same as "run" above, but include metrics generation and
#vcd generation
all: package run_emu report_metrics vcd

#Clean generated files
clean:
	@echo "Cleaning $(TARGET) Target Build Dir..."
	rm -rf $(BUILD_TARGET_DIR)

#Clean_all vivado reports...
clean_reports:
	rm -rf $(BLD_REPORTS_DIR)

#Clean_all xpes...
clean_xpe:
	rm -rf $(XPE_REPO)


TEST_COUNT := 0
# $(eval TEST_COUNT := $(shell expr $(TEST_COUNT) + 1 ))
test:
	TARGET=sw_emu EXTIO=0 TEST=1 GRAPH=argmax make clean_reports graph aiesim; \
	TARGET=sw_emu EXTIO=0 TEST=1 GRAPH=concat make clean_reports graph aiesim; \
	TARGET=sw_emu EXTIO=0 TEST=1 GRAPH=conv make clean_reports graph aiesim; \
	TARGET=sw_emu EXTIO=0 TEST=1 GRAPH=convchunk make clean_reports graph aiesim; \
	TARGET=sw_emu EXTIO=0 TEST=1 GRAPH=gemm make clean_reports graph aiesim; \
	TARGET=sw_emu EXTIO=0 TEST=1 GRAPH=identity make clean_reports graph aiesim; \
	TARGET=sw_emu EXTIO=0 TEST=1 GRAPH=pool make clean_reports graph aiesim; \
	TARGET=sw_emu EXTIO=0 TEST=1 GRAPH=transpose make clean_reports graph aiesim
