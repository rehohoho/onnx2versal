echo ""
date
echo ""
SECONDS=0

# Executing the elf...
export LD_LIBRARY_PATH=/mnt:/tmp:$LD_LIBRARY_PATH
export XCL_EMULATION_MODE=hw_emu
export XILINX_XRT=/usr
export XILINX_VITIS=/mnt
mkdir -p output/
./aie_xrt.elf a.xclbin 1 data output
python3 check.py -f1 data -f2 output

return_code=$?

if [ $return_code -ne 0 ]; then
  echo "ERROR: Embedded host run failed, RC=$return_code"
else
  echo "INFO: TEST PASSED, RC=0"
fi

duration=$SECONDS

echo ""
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
echo ""
date
echo ""
echo "INFO: Embedded host run completed."
echo ""

exit $return_code
