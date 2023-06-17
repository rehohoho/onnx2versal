## Profiling

(on aiesim)

GRAPH=tiny_ad

For float32 model run
```
python generate.py ../models/${GRAPH}.onnx ../data/$GRAPH/X_test.npy -ndata 10
TARGET=hw_emu GRAPH=${GRAPH} make graph aiesim_profile
```
For int8 model run
```
python -m onnxruntime.quantization.preprocess --input ../models/${GRAPH}.onnx --output ../models/${GRAPH}_infer.onnx
python quantize_onnx.py ../models/${GRAPH}_infer.onnx ../models/${GRAPH}_int8.onnx ../data/$GRAPH/X_test.npy
rm ../models/${GRAPH}_infer.onnx
python generate.py ../models/${GRAPH}_int8.onnx ../data/$GRAPH/X_test.npy
TARGET=hw_emu GRAPH=${GRAPH}_int8 make graph aiesim_profile
```

### hls4ml_jettag

```
# Truncated output
Running GemmReluMKKN<1,16,64>
start = 1634,end = 2041,total = 407
Running GemmReluMKKN<1,64,32>
start = 2454,end = 2964,total = 510
Running GemmReluMKKN<1,32,32>
start = 3335,end = 3633,total = 298
Running GemmMKKN<1,32,8>
start = 4004,end = 4123,total = 119
Running SoftmaxSingleaxis<1,5,8>
start = 4485,end = 4617,total = 132
```
total 1466 cycles

### hls4ml_jettag_int8

```
Running QuantizeLinearVector<1,16,16>
start = 1799,end = 1869,total = 70
Running QgemmVector<1,16,64>
start = 2045,end = 2215,total = 170
Running QgemmVector<1,64,32>
start = 2399,end = 2596,total = 197
Running QgemmVector<1,32,32>
start = 2768,end = 2901,total = 133
Running QgemmVector<1,32,16>
start = 3069,end = 3142,total = 73
Running QlinearsoftmaxSingleaxis<1,5,16>
start = 3310,end = 3480,total = 170
Running DequantizeLinearScalar<16,5>
start = 3660,end = 3724,total = 64
```
total 877 cycles


### lenet_mnist

```
# Truncated outoput
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
Running ConcatFloat<8,1,16,120>::filter8
start = 55936,end = 56062,total = 126
Running GemmReluMKKN<1,120,32>
Running GemmReluMKKN<1,120,32>
Running GemmReluMKKN<1,120,32>
start = 56382,end = 57270,total = 888
start = 56386,end = 57274,total = 888
start = 56390,end = 57278,total = 888
Running ConcatFloat<3,1,32,84>::filter3
start = 57426,end = 57499,total = 73
Running GemmReluMKKN<1,84,48>
start = 57771,end = 58744,total = 973
Running ConcatScalar<1,1,48,10>::filter1
start = 58888,end = 59766,total = 878
```
total 48037 cycles, note GemmReluMKKN runs in parallel

### lenet_mnist_int8

```
# Truncated output
Running QuantizeLinearVector<28,28,32>
start = 3035,end = 4980,total = 1945
Running QLinearConvVector<28,32,24,32,1,1,6,5>
start = 5372,end = 8609,total = 3237
Running Maxpool2x2Int8BCHW::filter<24,32,12,16,1,6>
start = 10137,end = 10461,total = 324
Running QLinearConvVector<12,16,8,16,1,6,16,5>
start = 10929,end = 15814,total = 4885
Running MaxpoolScalarBCHW::filter<8,16,4,4,1,16>
start = 16547,end = 20141,total = 3594
Running QgemmVector<1,256,128>
start = 20377,end = 22554,total = 2177
Running QgemmVector<1,128,96>
start = 22750,end = 23627,total = 877
Running QgemmVector<1,96,16>
start = 23827,end = 23976,total = 149
Running DequantizeLinearScalar<16,10>
start = 24140,end = 24222,total = 82
```
total 17270 cycles

### tiny_ad

```
Running GemmReluMKKNStream<14,640,128,0>
Running GemmReluMKKNStream<2,128,640,0>
[run tiny_ad]: success
Waiting for core(s) of graph tiny_ad to finish execution ...
start = 155332,end = 729472,total = 574140
Running MacFloat<f,14,128,1>
Running GemmReluMKKNStream<14,640,128,0>
start = 729612,end = 730569,total = 957
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
start = 732532,end = 740302,total = 7770
start = 732540,end = 740310,total = 7770
start = 732548,end = 740318,total = 7770
start = 732560,end = 740330,total = 7770
Running ConcatFloat<4,14,32,128>::filter4
start = 740479,end = 741284,total = 805
Running MacFloat<f,14,128,1>
start = 741440,end = 742397,total = 957
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
start = 744364,end = 752134,total = 7770
start = 744364,end = 752134,total = 7770
start = 744372,end = 752142,total = 7770
start = 744376,end = 752146,total = 7770
Running ConcatFloat<4,14,32,128>::filter4
start = 752295,end = 753100,total = 805
Running MacFloat<f,14,128,1>
start = 753256,end = 754213,total = 957
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
start = 756204,end = 763974,total = 7770
start = 756204,end = 763974,total = 7770
start = 756208,end = 763978,total = 7770
start = 756212,end = 763982,total = 7770
Running ConcatFloat<4,14,32,128>::filter4
start = 764131,end = 764935,total = 804
Running MacFloat<f,14,128,1>
start = 765091,end = 766230,total = 1139
Running GemmReluMKKN<14,128,8,0>
start = 766373,end = 771785,total = 5412
Running ConcatFloat<1,14,8,8>::filter7
start = 771928,end = 771969,total = 41
Running MacFloat<f,14,8,1>
start = 772113,end = 772276,total = 163
Running GemmReluMKKN<14,8,128,0>
start = 772419,end = 777123,total = 4704
Running ConcatFloat<1,14,128,128>::filter7
start = 777266,end = 777517,total = 251
Running MacFloat<f,14,128,1>
start = 779853,end = 780810,total = 957
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
start = 783172,end = 790944,total = 7772
start = 783184,end = 790956,total = 7772
start = 783168,end = 790966,total = 7798
start = 783168,end = 790967,total = 7799
Running ConcatFloat<4,14,32,128>::filter4
start = 791120,end = 791925,total = 805
Running MacFloat<f,14,128,1>
start = 794294,end = 795251,total = 957
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
start = 797568,end = 805338,total = 7770
start = 797574,end = 805344,total = 7770
start = 797580,end = 805350,total = 7770
start = 797586,end = 805356,total = 7770
Running ConcatFloat<4,14,32,128>::filter4
start = 805509,end = 806313,total = 804
Running MacFloat<f,14,128,1>
start = 808657,end = 809614,total = 957
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
Running GemmReluMKKN<14,128,32,0>
start = 811990,end = 819760,total = 7770
start = 812002,end = 819772,total = 7770
start = 812014,end = 819784,total = 7770
start = 812018,end = 819788,total = 7770
Running ConcatFloat<4,14,32,128>::filter4
start = 819933,end = 820738,total = 805
Running MacFloat<f,14,128,1>
start = 823133,end = 824090,total = 957
start = 155347,end = 992030,total = 836683
Running GemmReluMKKNStream<2,128,640,0>
start = 992187,end = 1160031,total = 167844
Running GemmReluMKKNStream<2,128,640,0>
start = 1160188,end = 1328032,total = 167844
Running GemmReluMKKNStream<2,128,640,0>
start = 1328189,end = 1496033,total = 167844
Running GemmReluMKKNStream<2,128,640,0>
start = 1496190,end = 1664034,total = 167844
Running GemmReluMKKNStream<2,128,640,0>
start = 1664191,end = 1832035,total = 167844
Running GemmReluMKKNStream<2,128,640,0>
start = 1832192,end = 2000036,total = 167844

Checking 1/9: k32mac_goldenout_shape14x128.txt                                                 against k32mac_goldenout_shape14x128.txt                 TEST: FAILED! Only 1790/1792 passed.
Max absolute difference: 0.00010728799999992766
Max relative difference: 0.007041693105805323
Checking 2/9: k35gemm_goldenout_shape14x128.txt                                                against k35gemm_goldenout_shape14x128.txt                TEST: FAILED! Only 1765/1792 passed.
Max absolute difference: 0.0004413119999999715
Max relative difference: 0.00788526848162412
Checking 3/9: k42mac_goldenout_shape14x128.txt                                                 against k42mac_goldenout_shape14x128.txt                 TEST: FAILED! Only 1789/1792 passed.
Max absolute difference: 6.93202000000559e-05
Max relative difference: 0.0043106067460254825
Checking 4/9: k27mac_goldenout_shape14x128.txt                                                 against k27mac_goldenout_shape14x128.txt                 TEST: FAILED! Only 1790/1792 passed.
Max absolute difference: 9.059899999996901e-05
Max relative difference: 0.007082397836386333
Checking 5/9: k25gemm_goldenout_shape14x128.txt                                                against k25gemm_goldenout_shape14x128.txt                TEST: FAILED! Only 1791/1792 passed.
Max absolute difference: 7.057200000026853e-05
Max relative difference: 0.016395304116482174
Checking 6/9: k30gemm_goldenout_shape14x128.txt                                                against k30gemm_goldenout_shape14x128.txt                TEST: FAILED! Only 1787/1792 passed.
Max absolute difference: 0.0002965930000016215
Max relative difference: 0.003946147511753102
Checking 7/9: k37mac_goldenout_shape14x128.txt                                                 against k37mac_goldenout_shape14x128.txt                 TEST: FAILED! Only 1789/1792 passed.
Max absolute difference: 8.189700000005296e-05
Max relative difference: 0.009867301893236126
Checking 8/9: k45gemm_goldenout_shape14x640.txt                                                against k45gemm_goldenout_shape14x640.txt                TEST: OK!
Checking 9/9: k40gemm_goldenout_shape14x128.txt                                                against k40gemm_goldenout_shape14x128.txt                TEST: FAILED! Only 1780/1792 passed.
Max absolute difference: 0.00022700429999999994
Max relative difference: 0.04030108283833806
```

### tiny_ad_int8

```
WARNING: scalebits 27 vs 22.678072 
WARNING: scalebits 27 vs 22.678072 
WARNING: scalebits 27 vs 22.678072 
WARNING: scalebits 27 vs 22.678072 
WARNING: scalebits 27 vs 22.678072 
WARNING: scalebits 27 vs 22.678072 
WARNING: scalebits 27 vs 22.678072 
WARNING: scalebits 27 vs 22.678072 
Running QuantizeLinearFmul<2,640,640>
start = 4587,end = 7028,total = 2441
Running QuantizeLinearFmul<2,640,640>
start = 7191,end = 9632,total = 2441
Running QuantizeLinearFmul<2,640,640>
start = 9795,end = 12236,total = 2441
Running QuantizeLinearFmul<2,640,640>
start = 12399,end = 14840,total = 2441
Running QuantizeLinearFmul<2,640,640>
start = 15003,end = 17444,total = 2441
Running QuantizeLinearFmul<2,640,640>
start = 17607,end = 20048,total = 2441
Running QuantizeLinearFmul<2,640,640>
start = 20211,end = 22652,total = 2441
Running QgemmVector<14,640,16>
Running QgemmVector<14,640,16>
Running QgemmVector<14,640,16>
Running QgemmVector<14,640,16>
Running QgemmVector<14,640,16>
Running QgemmVector<14,640,16>
Running QgemmVector<14,640,16>
Running QgemmVector<14,640,16>
start = 23148,end = 32514,total = 9366
start = 23152,end = 32518,total = 9366
start = 23160,end = 32535,total = 9375
start = 23168,end = 32535,total = 9367
start = 23176,end = 32542,total = 9366
start = 23180,end = 32546,total = 9366
start = 23188,end = 32567,total = 9379
start = 23156,end = 32570,total = 9414
Running ConcatScalar<a,8,14,16,128>::filter8
start = 32821,end = 45833,total = 13012
Running QlinearMac<14,128,0>
start = 46001,end = 47253,total = 1252
Running QgemmVector<14,128,128>
start = 47397,end = 63119,total = 15722
Running QlinearMac<14,128,0>
start = 63263,end = 64515,total = 1252
Running QgemmVector<14,128,128>
start = 64659,end = 80381,total = 15722
Running QlinearMac<14,128,0>
start = 80525,end = 81777,total = 1252
Running QgemmVector<14,128,128>
start = 81921,end = 97643,total = 15722
Running QlinearMac<14,128,0>
start = 97787,end = 99039,total = 1252
Running QgemmVector<14,128,16>
start = 99183,end = 101379,total = 2196
Running QlinearMac<14,16,0>
start = 101523,end = 101722,total = 199
Running QgemmVector<14,16,128>
start = 101866,end = 105984,total = 4118
Running QlinearMac<14,128,0>
start = 106128,end = 107380,total = 1252
Running QgemmVector<14,128,128>
start = 108073,end = 123795,total = 15722
Running QlinearMac<14,128,0>
start = 124471,end = 125723,total = 1252
Running QgemmVector<14,128,128>
start = 126398,end = 142310,total = 15912
Running QlinearMac<14,128,0>
start = 142966,end = 144218,total = 1252
Running QgemmVector<14,128,128>
start = 144883,end = 160605,total = 15722
Running QlinearMac<14,128,0>
start = 161234,end = 162486,total = 1252
Running QgemmVector<14,128,128>
Running QgemmVector<14,128,128>
Running QgemmVector<14,128,128>
Running QgemmVector<14,128,128>
Running QgemmVector<14,128,128>
start = 163121,end = 178843,total = 15722
start = 163125,end = 178847,total = 15722
start = 163129,end = 178851,total = 15722
start = 163139,end = 178861,total = 15722
start = 163151,end = 178872,total = 15721
Running ConcatScalar<a,5,14,128,640>::filter5
start = 179467,end = 243044,total = 63577
Running DequantizeLinear<2,640,640>
start = 243546,end = 245150,total = 1604
Running DequantizeLinear<2,640,640>
start = 245313,end = 246917,total = 1604
Running DequantizeLinear<2,640,640>
start = 247088,end = 248693,total = 1605
Running DequantizeLinear<2,640,640>
start = 248856,end = 250460,total = 1604
Running DequantizeLinear<2,640,640>
start = 250628,end = 252232,total = 1604
Running DequantizeLinear<2,640,640>
start = 252395,end = 253999,total = 1604
Running DequantizeLinear<2,640,640>
start = 254162,end = 255766,total = 1604


Checking 1/9: k17qlinearmac_goldenout_shape14x128.txt                                          against k17qlinearmac_goldenout_shape14x128.txt          TEST: FAILED! Only 1785/1792 passed.
Max absolute difference: 4.0
Max relative difference: 0.036036036036036036
Checking 2/9: k20qlinearmac_goldenout_shape14x128.txt                                          against k20qlinearmac_goldenout_shape14x128.txt          TEST: FAILED! Only 1766/1792 passed.
Max absolute difference: 10.0
Max relative difference: 0.0975609756097561
Checking 3/9: k29dequantizeLinear_goldenout_shape14x640.txt                                    against k29dequantizeLinear_goldenout_shape14x640.txt    TEST: FAILED! Only 8491/8960 passed.
Max absolute difference: 2.251219749999999
Max relative difference: 0.22222841200353474
Checking 4/9: k19qgemm_goldenout_shape14x128.txt                                               against k19qgemm_goldenout_shape14x128.txt               TEST: FAILED! Only 1738/1792 passed.
Max absolute difference: 4.0
Max relative difference: 1.0
Checking 5/9: k22qgemm_goldenout_shape14x128.txt                                               against k22qgemm_goldenout_shape14x128.txt               TEST: FAILED! Only 1719/1792 passed.
Max absolute difference: 7.0
Max relative difference: 2.5
Checking 6/9: k28qgemm_goldenout_shape14x640.txt                                               against k28qgemm_goldenout_shape14x640.txt               TEST: FAILED! Only 8491/8960 passed.
Max absolute difference: 6.0
Max relative difference: 3.0
Checking 7/9: k23qlinearmac_goldenout_shape14x128.txt                                          against k23qlinearmac_goldenout_shape14x128.txt          TEST: FAILED! Only 1762/1792 passed.
Max absolute difference: 11.0
Max relative difference: 0.2
Checking 8/9: k25qgemm_goldenout_shape14x128.txt                                               against k25qgemm_goldenout_shape14x128.txt               TEST: FAILED! Only 1697/1792 passed.
Max absolute difference: 6.0
Max relative difference: 2.5
Checking 9/9: k26qlinearmac_goldenout_shape14x128.txt                                          against k26qlinearmac_goldenout_shape14x128.txt          TEST: FAILED! Only 1721/1792 passed.
Max absolute difference: 13.0
Max relative difference: 0.15853658536585366
```

convchunk test
```
Running Pad2DScalar::filter<f,1,24,24,2,2,2,2>
Running SplitScalar<f,2,1,784,448>::filter2
start = 824,end = 13480,total = 12656
start = 857,end = 13868,total = 13011
Running ConvReluScalarStreamCacheCKK<16,28,24,1,1,1,1,5,5,1>
Running ConvReluScalarStreamCacheCKK<16,28,24,1,1,1,1,5,5,1>
start = 14018,end = 149269,total = 135251
start = 14022,end = 149273,total = 135251
Running ConcatScalar<f,2,5,288,576>::filter2
if tout.nbytes > MAX_PARAM_SIZE:
start = 149338,end = 181050,total = 31712
```