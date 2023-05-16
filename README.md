# onnx2versal

Random notes that I haven't organized.
* Trafficgen for actual graphs only.
* Profiling cycles for trafficgen is disabled.
* LOG_PROFILE=1 to enable profiling information and outputting intermediates.
* Outputting intermediates is limited to <=7 due to <=8 cascade channels
* Use only generated files to unit test kernels.
* Kernels with parameters must be vector readable (128-bit chunks), pad as required
  * Gemm: KxN, N must be 128-bit chunks, if type=float32, N%4=0

hw_emu runs real slow if cycle count is high, don't bother

~500000 - 900000 samples/s on hardware

## Kernel Example

Run tests
```
# X86 GRAPH
TARGET=sw_emu EXTIO=0 GRAPH=convchunk LOG_PROFILE=1 make clean_reports graph aiesim

# SYSC GRAPH
TARGET=hw_emu EXTIO=0 GRAPH=convchunk LOG_PROFILE=1 make clean_reports graph aiesim
```

## Write a Kernel
TBC

## Write a Graph
TBC 
