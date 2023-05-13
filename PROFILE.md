Gemm:
params: 
ins:    in     (MK), weight (NK), bias (N)
outs:   out    (MN)
  Running GemmReluScalarGmemParamMKNK<1, 86, 10> total = 19220

params: weight (NK), bias (N)
ins:    in     (MK)
outs:   out    (MN)
  Running GemmReluScalarMKNK<1, 86, 10> total = 1939

params: weight (KN), bias (N)
ins:    in     (MK)
outs:   out    (MN)
  Running GemmReluScalarMKKN<1, 86, 10> total = 
  Running GemmReluMKKN<1, 86, 10> total = 366


Conv:
params: weights (MCKK), bias (M)
ins:    in      (BCHW)
outs:   out     (BMHW)
  Running ConvReluScalarBCHW<28, 24, 1, 2, 2, 5> total = 148375  
  Running Conv5x5ReluBCHW<28, 24, 1, 2, 2> total = 16241
  Running Conv5x5on8ReluBCHW<28, 24, 1, 2, 2> total = 10687

params: weights (MKKC), bias (M)
ins:    in      (BHWC)
outs:   out     (BHWM)
  Running ConvReluScalarBHWC<28, 24, 1, 2, 2, 5> total = 144244
  
params:
ins:    in      (BHWC), weight (MKKC), bias (M)
outs:   out     (BHWM)
  Running ConvReluScalarGmemParamBHWC<28, 24, 1, 2, 2, 5> total = 1382714


Pool:
Running MaxpoolScalarBHWC::filter<24, 12, 1, 6> total = 7673
Running MaxpoolScalarBCHW::filter<24, 12, 1, 6> total = 11302
Running Maxpool2x2BCHW::filter<24, 12, 1, 6> total = 901


Concat:
ConcatScalar<LCNT, WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>
ConcatVector<LCNT, WINDOW_SIZE, CHUNK_SIZE, BLOCK_SIZE>: CHUNK_SIZE%8=0, BLOCK_SIZE%4=0
Concatenates chunks of CHUNK_SIZE from LCNT lanes then truncate to BLOCK_SIZE, outputs size of WINDOW_SIZE / CHUNK_SIZE * BLOCK_SIZE
  LCNT:         number of lanes to concat
  WINDOW_SIZE:  size of window for each lane
  CHUNK_SIZE:   size of chunk from each lanes per iteration
  BLOCK_SIZE:   size of concatenated chunks per iteration

    Running ConcatScalar<64, 16, 52>::filter5, total = 650
    Running ConcatVector<64, 16, 52>::filter5, total = 232

Implementation notes:
  Using virtual function instead of macro has big overhead: 163 -> 1047