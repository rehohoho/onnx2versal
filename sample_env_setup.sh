#!/bin/bash

# =======================================================
# Set Platform ,Vitis and Versal Image repo
# =======================================================
export PLATFORM_REPO_PATHS=/tools/Xilinx/Vitis/2022.2/base_platforms
export XILINX_VITIS=/tools/Xilinx/Vitis/2022.2
export COMMON_IMAGE_VERSAL=/tools/xilinx-versal-common-v2022.2
export XILINX_X86_XRT=/opt/xilinx/xrt
export PYTHON3_LOCATION=/usr/bin

# ====================================================
# Source Versal Image ,Vitis and Aietools
# ====================================================
# Run the below command to setup environment and CXX
source $COMMON_IMAGE_VERSAL/environment-setup-cortexa72-cortexa53-xilinx-linux
source $XILINX_VITIS/settings64.sh

# =========================================================
# Platform Selection...
# =========================================================
tgt_plat=xilinx_vck190_base_202220_1
export PLATFORM=$PLATFORM_REPO_PATHS/$tgt_plat/$tgt_plat\.xpfm

# ==========================================================
# Validating Tool Installation
# ==========================================================
echo "Aiecompiler:    $(which aiecompiler)"
echo "Vivado:         $(which vivado)"
echo "Vitis:          $(which vitis)"
echo "Vitis HLS:      $(which vitis_hls)"
echo "XILINX_X86_XRT: ${XILINX_X86_XRT}"

# Python setup
export PYTHONPATH=${XILINX_VIVADO}/data/emulation/hw_em/lib/python/:${XILINX_VIVADO}/data/emulation/ip_utils/xtlm_ipc/xtlm_ipc_v1_0/python/:$XILINX_VIVADO/data/emulation/python/xtlm_ipc:${PYTHONPATH}
export PATH=$PYTHON3_LOCATION:$PATH