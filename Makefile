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
	@echo  "    TARGET             sw_emu (default) | hw_emu | hw,  build target"
	@echo  "    EXTIO              true (default) | false,          traffic gen usage"
	@echo  "    NET_INSTS          1 (default) | 5 | 50"
	@echo  "    PL_FREQ            250Mhz (default),                HLS kernel frequencies"
	@echo  "    ITER_CNT           8 (default),                     design iteration count"
	@echo  "    EN_TRACE           0 (default) | 1,                 enable profiling .ini (hw)"
	@echo  ""
	@echo  " Runs in x86simulator / x86 threads. Functional only, no profiling."
	@echo  "  TARGET=sw_emu EXTIO=false make kernels graph aiesim"
	@echo  "  TARGET=sw_emu EXTIO=false make kernels graph xsa application package run_emu"
	@echo  "  TARGET=sw_emu EXTIO=true make kernels graph aiesim"
	@echo  "  TARGET=sw_emu EXTIO=true make kernels graph xsa application package run_emu"
	@echo  "  TARGET=sw_emu make clean clean_reports clean_xpe"
	@echo  ""
	@echo  " Runs in aiesimulator / QEMU. Profiling ok."
	@echo  "  TARGET=hw_emu EXTIO=false make kernels graph aiesim"
	@echo  "  TARGET=hw_emu EXTIO=false make kernels graph xsa application package run_emu"
	@echo  "  TARGET=hw_emu EXTIO=true make kernels graph aiesim"
	@echo  "  TARGET=hw_emu EXTIO=true make kernels graph xsa application package run_emu"
	@echo  "  TARGET=hw_emu make clean clean_reports clean_xpe"
	@echo  ""
	@echo  " Other build options: aiesim_profile, vcd, run, all, report_metrics"
	@echo  ""

# Print all options passed to Makefile
print-%  : ; @echo $* = $($*)

# =========================================================
# TARGET can be set as:
#   hw_emu: Hardware Emulation
#   hw    : Hardware Run
# =========================================================
TARGET ?= sw_emu
EXTIO ?= true

# =========================================================
# No of Instances can be set as:
#   1(default).
# =========================================================
NET_INSTS := 1

# =========================================================
# No of Design Iterations 
#   1(default).
# =========================================================
ITER_CNT := 1

# =========================================================
# PL Frequency in Mhz:
#   312.5Mhz(default)
# =========================================================
PL_FREQ := 312.5
HZ_UNIT := 1000000
VPP_CLOCK_FREQ := $(shell printf "%.0f" `echo "${PL_FREQ} * $(HZ_UNIT)" | bc`)
#VPP_CLOCK_FREQ := $(PL_FREQ)000000

# =========================================================
# Profiling/Trace and Datamover type Switches...
# =========================================================
EN_TRACE	:= 0

############################################################
# maximum cycle count for aiesimulator to get powerfrom vcd
# default 100
# #########################################################
MAX_CYCLES   := 100000
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

BASE_BLD_DIR     	:= $(PROJECT_REPO)/build
INSTS_BLD_DIR     := $(BASE_BLD_DIR)/net_x$(NET_INSTS)
BUILD_TARGET_DIR  := $(INSTS_BLD_DIR)/$(TARGET)

REPORTS_REPO := $(PROJECT_REPO)/reports_dir
BLD_REPORTS_DIR := $(REPORTS_REPO)/net_x$(NET_INSTS)/$(TARGET)

XPE_REPO         := $(PROJECT_REPO)/xpe_dir
BLD_XPE_DIR      := $(XPE_REPO)/net_x$(NET_INSTS)
VCD_FILE_NAME    := net_x$(NET_INSTS)
BLD_TGT_VCD_FILE := $(BUILD_TARGET_DIR)/$(VCD_FILE_NAME).vcd
XPE_FILE         := $(BLD_XPE_DIR)/graph_$(VCD_FILE_NAME).xpe

EMBEDDED_PACKAGE_OUT := $(BUILD_TARGET_DIR)/package
EMBEDDED_EXEC_SCRIPT := run_script.sh

WORK_DIR := Work

###########################################################
# Variable Definitions...
#
# ==========================================================
# Below are the names for SDF graph, application executable,
# kernel executables, and xsa
# ==========================================================

# =========================================================
# Kernel Source Files repository
# =========================================================

# =========================================================
# Graph Source files repository
# ========================================================

# =========================================================
# Application Source Files repository
# =========================================================

# =========================================================
# AIE Compiler Global Settings and Include Libraries
# =========================================================
# =========================================================
# Application Compiler and Linker Flags
# =========================================================

# =========================================================
# Kernel Compiler and Linker Flags
# ========================================================

# =========================================================
# Packaging Flags
# ========================================================

# =========================================================
# Step 1. Kernel XO File Generation
# ========================================================
# This step compiles the HLS C PL kernels. 
# Outputs: in build/[hw_emu | hw]/ directory
# 	dma_hls.[hw_emu | hw].xo  
#	dma_hls.[hw_emu | hw].xo.compile_summary  
#	v++_dma_hls.[hw_emu | hw].log
#	_x
DATAMOVER_KERNEL_TOP := dma_hls
DATAMOVER_KERNEL_XO  := $(DATAMOVER_KERNEL_TOP).$(TARGET)

TRAFFICGEN_WIDTH := 32

DATAMOVER_KERNEL_VPP_FLAGS := --hls.clock $(VPP_CLOCK_FREQ):$(DATAMOVER_KERNEL_TOP)

VPP_FLAGS := --platform $(PLATFORM) \
					   --save-temps \
					   --temp_dir $(BUILD_TARGET_DIR)/_x \
					   --verbose \
					   -g

DATAMOVER_KERNEL_SRC := $(PL_SRC_REPO)/datamover/$(DATAMOVER_KERNEL_TOP).cpp

ifeq ($(EXTIO), true)
	KERNEL_XOS := $(XILINX_VITIS)/data/emulation/XO/sim_ipc_axis_master_$(TRAFFICGEN_WIDTH).xo \
								$(XILINX_VITIS)/data/emulation/XO/sim_ipc_axis_slave_$(TRAFFICGEN_WIDTH).xo
else
	KERNEL_XOS := $(BUILD_TARGET_DIR)/$(DATAMOVER_KERNEL_XO).xo
endif

kernels: $(KERNEL_XOS)

$(BUILD_TARGET_DIR)/$(DATAMOVER_KERNEL_XO).xo:
	mkdir -p $(BUILD_TARGET_DIR); \
	cd $(BUILD_TARGET_DIR); \
	v++ --target $(TARGET) $(DATAMOVER_KERNEL_VPP_FLAGS) \
		$(VPP_FLAGS) -c -k $(DATAMOVER_KERNEL_TOP) \
		$(DATAMOVER_KERNEL_SRC) -o $@

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
GRAPH_SRC_CPP := $(AIE_SRC_REPO)/graph_lenet.cpp

AIE_FLAGS := -include=$(AIE_SRC_REPO) \
						 --verbose \
						 --Xpreproc="-DNET_INSTS=$(NET_INSTS)" \
						 --Xpreproc="-DITER_CNT=$(ITER_CNT)" \
						 --Xchess="main:backend.mist2.maxfoldk=256" \
						 --platform=$(PLATFORM) \
						 --log-level=5 \
						 --pl-freq=500 \
						 --dataflow \
						 --heapsize=2048 \
						 --workdir=$(WORK_DIR)
ifeq ($(TARGET), sw_emu)
	AIE_FLAGS += --target=x86sim --Xpreproc=-O0
endif
ifeq ($(EXTIO), true)
	AIE_FLAGS += --Xpreproc=-DEXTERNAL_IO
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

TRAFFIC_GEN_PY := $(DESIGN_REPO)/trafficgen/xtg_lenet.py

aiesim: graph
ifeq ($(EXTIO), false)
ifeq ($(TARGET), sw_emu)
	mkdir -p $(X86SIM_REPORT_DIR); \
	cd $(BUILD_TARGET_DIR); \
	x86simulator $(AIE_SIM_FLAGS) -o=$(X86SIM_REPORT_DIR) 2>&1 | tee -a $(X86SIM_REPORT_DIR)/x86sim.log
else
	mkdir -p $(AIESIM_REPORT_DIR); \
	cd $(BUILD_TARGET_DIR); \
	aiesimulator $(AIE_SIM_FLAGS) -o $(AIESIM_REPORT_DIR) 2>&1 | tee -a $(AIESIM_REPORT_DIR)/aiesim.log
endif
else  # Use External Traffic Generator
ifeq ($(TARGET), sw_emu)
	mkdir -p $(X86SIM_REPORT_DIR); \
	cd $(BUILD_TARGET_DIR); \
	x86simulator $(AIE_SIM_FLAGS) -o=$(X86SIM_REPORT_DIR) 2>&1 | tee -a $(X86SIM_REPORT_DIR)/x86sim.log & python3 $(TRAFFIC_GEN_PY) --input_dir $(DATA_REPO) --output_dir $(X86SIM_REPORT_DIR) 2>&1 | tee -a $(X86SIM_REPORT_DIR)/x86sim_filetraffic.log
else
	mkdir -p $(AIESIM_REPORT_DIR); \
	cd $(BUILD_TARGET_DIR); \
	aiesimulator $(AIE_SIM_FLAGS) -o $(AIESIM_REPORT_DIR) 2>&1 | tee -a $(AIESIM_REPORT_DIR)/aiesim.log & python3 $(TRAFFIC_GEN_PY) --input_dir $(DATA_REPO) --output_dir $(AIESIM_REPORT_DIR) 2>&1 | tee -a $(AIESIM_REPORT_DIR)/aiesim_filetraffic.log
endif
endif

aiesim_profile: graph
ifeq ($(EXTIO), false)
ifeq ($(TARGET), sw_emu)
	mkdir -p $(X86SIM_REPORT_DIR); \
	cd $(BUILD_TARGET_DIR); \
	x86simulator $(AIE_SIM_FLAGS) -o=$(X86SIM_REPORT_DIR) 2>&1 | tee -a $(X86SIM_REPORT_DIR)/x86sim.log
else
	mkdir -p $(AIESIM_REPORT_DIR); \
	cd $(BUILD_TARGET_DIR); \
	aiesimulator $(AIE_SIM_FLAGS) --profile -o $(AIESIM_REPORT_DIR) 2>&1 | tee -a $(AIESIM_REPORT_DIR)/aiesim.log
endif
else  # Use External Traffic Generator
ifeq ($(TARGET), sw_emu)
	mkdir -p $(X86SIM_REPORT_DIR); \
	cd $(BUILD_TARGET_DIR); \
	x86simulator $(AIE_SIM_FLAGS) -o=$(X86SIM_REPORT_DIR) 2>&1 | tee -a $(X86SIM_REPORT_DIR)/x86sim.log & python3 $(TRAFFIC_GEN_PY) --input_dir $(DATA_REPO) --output_dir $(X86SIM_REPORT_DIR) 2>&1 | tee -a $(X86SIM_REPORT_DIR)/x86sim_filetraffic.log
else
	mkdir -p $(AIESIM_REPORT_DIR); \
	cd $(BUILD_TARGET_DIR); \
	aiesimulator $(AIE_SIM_FLAGS) --profile -o $(AIESIM_REPORT_DIR) 2>&1 | tee -a $(AIESIM_REPORT_DIR)/aiesim.log & python3 $(TRAFFIC_GEN_PY) --input_dir $(DATA_REPO) --output_dir $(AIESIM_REPORT_DIR) 2>&1 | tee -a $(AIESIM_REPORT_DIR)/aiesim_filetraffic.log
endif
endif

vcd: graph $(XPE_FILE)

# xpe file generation...
$(XPE_FILE): $(BLD_TGT_VCD_FILE)
	cd $(BUILD_TARGET_DIR); \
	vcdanalyze --vcd $(VCD_FILE_NAME).vcd --xpe
	rm -rf $(BLD_XPE_DIR)
	mkdir -p $(BLD_XPE_DIR)
	cp -rf $(BUILD_TARGET_DIR)/aiesim_xpe/*.xpe $(XPE_FILE)

# vcd file generation...
$(BLD_TGT_VCD_FILE):
	cd $(BUILD_TARGET_DIR); \
	aiesimulator $(AIE_SIM_FLAGS) --dump-vcd $(VCD_FILE_NAME) 2>&1 | tee -a vcd.log

# =========================================================
# Step 3. XSA File Generation
# ========================================================
# This step links the graph executable (tx_chain.o) and 
# the kernels into a xsa file. 
# Outputs: in build/[hw_emu | hw]/ directory
APP_OBJ_NAME := net_aie
XSA := vck190_$(APP_OBJ_NAME).xsa

VPP_LINK_FLAGS := --clock.defaultTolerance 0.001 \
                  --advanced.param compiler.userPostSysLinkOverlayTcl=$(DIRECTIVES_REPO)/noc_qos.tcl \
                  --vivado.prop run.synth_1.STEPS.SYNTH_DESIGN.ARGS.CONTROL_SET_OPT_THRESHOLD=16

ifeq ($(EXTIO), true)
	VPP_LINK_FLAGS += --config $(SYSTEM_CONFIGS_REPO)/mul_etg.cfg
else
	VPP_LINK_FLAGS += --clock.freqHz $(VPP_CLOCK_FREQ):$(DATAMOVER_KERNEL_TOP)_0 \
										--config $(SYSTEM_CONFIGS_REPO)/mul.cfg
endif

ifeq ($(EN_TRACE),1)
   ifeq ($(TARGET),hw)
      VPP_LINK_FLAGS += --profile.data $(DATAMOVER_KERNEL_TOP):all:all \
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

$(BUILD_TARGET_DIR)/$(XSA):$(KERNEL_XOS) $(LIBADF_A) $(SYSTEM_CONFIGS_REPO)/*
	cd $(BUILD_TARGET_DIR);	\
	v++ -l $(VPP_FLAGS) $(VPP_LINK_FLAGS) -t $(TARGET) -o $@ \
		    $(KERNEL_XOS) $(LIBADF_A)

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
APP_ELF 				:= $(APP_OBJ_NAME)_xrt.elf
APP_SRC_CPP 	  := $(HOST_APP_SRC_REPO)/net_aie_app.cpp
AIE_CONTROL_CPP := $(BUILD_TARGET_DIR)/$(WORK_DIR)/ps/c_rts/aie_control_xrt.cpp

GCC_FLAGS := -O \
             -c \
             -std=c++17 \
             -D__linux__ \
             -D__PS_ENABLE_AIE__ \
             -DXAIE_DEBUG \
             -DITER_CNT=$(ITER_CNT) \
             -DNET_INSTS=$(NET_INSTS)

GCC_INC_FLAGS := -I$(XILINX_VITIS)/aietools/include/ \
                 -I$(AIE_SRC_REPO) \
                 -I$(HOST_APP_SRC_REPO)

GCC_INC_LIB := -ladf_api_xrt \
               -lxrt_coreutil 

ifeq ($(EXTIO), true)
GCC_FLAGS += -DEXTERNAL_IO
endif

ifeq ($(TARGET), sw_emu)
CXX := g++
GCC_FLAGS += -D__SYNCBO_ENABLE__
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

$(BUILD_TARGET_DIR)/$(APP_ELF): $(HOST_APP_SRC)/* $(LIBADF_A)
	@rm -rf $(BUILD_TARGET_DIR)/app_control.o $(BUILD_TARGET_DIR)/$(APP_OBJ_NAME)_app.o $(BUILD_TARGET_DIR)/$(APP_ELF)
	$(CXX) $(GCC_FLAGS) $(GCC_INC_FLAGS) $(AIE_CONTROL_CPP) -o $(BUILD_TARGET_DIR)/app_control.o
	$(CXX) $(GCC_FLAGS) $(GCC_INC_FLAGS) $(APP_SRC_CPP) -o $(BUILD_TARGET_DIR)/$(APP_OBJ_NAME)_app.o $(GCC_INC_LIB)
	$(CXX) $(BUILD_TARGET_DIR)/app_control.o $(BUILD_TARGET_DIR)/$(APP_OBJ_NAME)_app.o $(GCC_INC_LIB) -o $(BUILD_TARGET_DIR)/$(APP_ELF)

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
PKG_FLAGS := -t $(TARGET) \
             --save-temps \
             --temp_dir $(BUILD_TARGET_DIR)/_x \
             -f $(PLATFORM) \
						 --package.defer_aie_run \
						 $(BUILD_TARGET_DIR)/$(XSA) $(LIBADF_A)

ifeq ($(TARGET), sw_emu)
PKG_FLAGS += -o $(BUILD_TARGET_DIR)/a.xclbin
else
PKG_FLAGS += --package.rootfs $(COMMON_IMAGE_VERSAL)/rootfs.ext4 \
             --package.kernel_image $(COMMON_IMAGE_VERSAL)/Image \
             --package.boot_mode=sd \
             --package.out_dir $(EMBEDDED_PACKAGE_OUT) \
             --package.image_format=ext4 \
             --package.sd_file $(BUILD_TARGET_DIR)/$(APP_ELF) \
             --package.sd_file $(EXEC_SCRIPTS_REPO)/$(EMBEDDED_EXEC_SCRIPT)
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

$(EMBEDDED_PACKAGE_OUT): $(PROFILING_CONFIGS_REPO)/* $(EXEC_SCRIPTS_REPO)/* $(ALL_APP_ELFS)
	rm -rf $(EMBEDDED_PACKAGE_OUT)
	cd $(BUILD_TARGET_DIR);	\
	v++ -p $(PKG_FLAGS)
	#cp -rf $(EMBEDDED_PACKAGE_OUT) $(EMBEDDED_PACKAGE_OUT)_backup

# =========================================================
# Step 6. Run Hardware Emulation 
# ========================================================
# If the target is for HW_EMU, launch the emulator
# If the target is for HW, you'll have to follow the
# instructions in the README.md
run_emu:
ifeq ($(TARGET),hw_emu)

ifeq ($(EXTIO), true)
	mkdir -p $(BLD_REPORTS_DIR)/aiesimulator_output; \
	cd $(EMBEDDED_PACKAGE_OUT); \
	python3 $(TRAFFIC_GEN_PY) --input_dir $(DATA_REPO) --output_dir $(BLD_REPORTS_DIR)/aiesimulator_output 2>&1 | tee embedded_run_trafficgen.log & \
	./launch_hw_emu.sh -run-app $(EMBEDDED_EXEC_SCRIPT) -no-reboot | tee embedded_run.log
else
	cd $(EMBEDDED_PACKAGE_OUT); \
	./launch_hw_emu.sh -run-app $(EMBEDDED_EXEC_SCRIPT) -no-reboot | tee embedded_run.log
endif

else # sw_emu

ifeq ($(EXTIO), true)
	mkdir -p $(BLD_REPORTS_DIR)/x86simulator_output; \
	cd $(BUILD_TARGET_DIR); \
	export XILINX_XRT=$(XILINX_X86_XRT); \
	export XCL_EMULATION_MODE=sw_emu; \
	export LD_LIBRARY_PATH=${XILINX_X86_XRT}/lib; \
	./$(APP_ELF) a.xclbin 2>&1 | tee embedded_run.log & \
	python3 $(TRAFFIC_GEN_PY) --input_dir $(DATA_REPO) --output_dir $(BLD_REPORTS_DIR)/x86simulator_output 2>&1 | tee embedded_run_trafficgen.log; \
	unset LD_LIBRARY_PATH
else
	@echo "sw_emu without EXTIO is not possible since kernel requires hw | hw_emu"
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
