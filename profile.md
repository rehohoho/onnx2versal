## Profiling

(on aiesim)

GRAPH=tiny_vww

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
QuantizeLinearFmulStream<1,16,16> start = 920,end = 989,total = 69
QgemmStream<1,16,64> start = 1001,end = 1182,total = 181
QgemmStream<1,64,32> start = 1196,end = 1398,total = 202
QgemmStream<1,32,32> start = 1412,end = 1553,total = 141
QgemmStream<1,32,16> start = 1567,end = 1650,total = 83
QlinearsoftmaxSingleaxis<1,5,16> start = 1664,end = 1835,total = 171
Running DequantizeLinear<1,16,8> start = 1910,end = 1960,total = 50
```
start to end 1040 cycles


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
Running QLinearConv5x5<28,32,24,32,1,1,6,5>
start = 5372,end = 8609,total = 3237
Running Maxpool2x2Int8BCHW::filter<24,32,12,16,1,6>
start = 10137,end = 10461,total = 324
Running QLinearConv5x5<12,16,8,16,1,6,16,5>
start = 10929,end = 15814,total = 4885
Running MaxpoolScalarBCHW::filter<8,16,4,4,1,16>
start = 16547,end = 20141,total = 3594
Running QgemmStream<1,256,128>
start = 20377,end = 22554,total = 2177
Running QgemmStream<1,128,96>
start = 22750,end = 23627,total = 877
Running QgemmStream<1,96,16>
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
Running QgemmStream<14,640,16>
Running QgemmStream<14,640,16>
Running QgemmStream<14,640,16>
Running QgemmStream<14,640,16>
Running QgemmStream<14,640,16>
Running QgemmStream<14,640,16>
Running QgemmStream<14,640,16>
Running QgemmStream<14,640,16>
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
Running QgemmStream<14,128,128>
start = 47397,end = 63119,total = 15722
Running QlinearMac<14,128,0>
start = 63263,end = 64515,total = 1252
Running QgemmStream<14,128,128>
start = 64659,end = 80381,total = 15722
Running QlinearMac<14,128,0>
start = 80525,end = 81777,total = 1252
Running QgemmStream<14,128,128>
start = 81921,end = 97643,total = 15722
Running QlinearMac<14,128,0>
start = 97787,end = 99039,total = 1252
Running QgemmStream<14,128,16>
start = 99183,end = 101379,total = 2196
Running QlinearMac<14,16,0>
start = 101523,end = 101722,total = 199
Running QgemmStream<14,16,128>
start = 101866,end = 105984,total = 4118
Running QlinearMac<14,128,0>
start = 106128,end = 107380,total = 1252
Running QgemmStream<14,128,128>
start = 108073,end = 123795,total = 15722
Running QlinearMac<14,128,0>
start = 124471,end = 125723,total = 1252
Running QgemmStream<14,128,128>
start = 126398,end = 142310,total = 15912
Running QlinearMac<14,128,0>
start = 142966,end = 144218,total = 1252
Running QgemmStream<14,128,128>
start = 144883,end = 160605,total = 15722
Running QlinearMac<14,128,0>
start = 161234,end = 162486,total = 1252
Running QgemmStream<14,128,128>
Running QgemmStream<14,128,128>
Running QgemmStream<14,128,128>
Running QgemmStream<14,128,128>
Running QgemmStream<14,128,128>
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

### tiny_ic
```
```

### tiny_ic_int8
```
TransposeScalarBHWC2BCHW<f,1,32,32,3> start = 6835,end = 13230,total = 6395
QuantizeLinearFmulStream<96,32,32> start = 6804,end = 19522,total = 12718
Pad2DStreamInt8<a,3,32,32,1,1,1,15> start = 6802,end = 19605,total = 12803
QLinearConvHx4Stream<34,48,32,32,1,1,1,3,16,3> start = 19626,end = 37963,total = 18337
Pad2DStreamInt8<a,16,32,32,1,1,1,15> start = 6805,end = 44724,total = 37919
SplitInt8<a,16,1632,864,96>::filter2 start = 6840,end = 44760,total = 37920
QLinearConvHx4Stream<18,48,32,32,1,1,1,16,16,3> start = 44892,end = 82869,total = 37977
QLinearConvHx4Stream<18,48,32,32,1,1,1,16,16,3> start = 44896,end = 82873,total = 37977
ConcatInt8Stream<a,16,512,512,1024> start = 6808,end = 83007,total = 76199
Pad2DStreamInt8<a,16,32,32,1,1,1,15> start = 6813,end = 83719,total = 76906
SplitInt8<a,16,1632,864,96>::filter2 start = 6858,end = 83760,total = 76902
QLinearConvHx4Stream<18,48,32,32,1,1,1,16,16,3> start = 83892,end = 121869,total = 37977
QLinearConvHx4Stream<18,48,32,32,1,1,1,16,16,3> start = 83896,end = 121873,total = 37977
ConcatInt8Stream<a,16,512,512,1024> start = 6826,end = 122007,total = 115181
QLinearAddInt8<a,16384,0> start = 6888,end = 122164,total = 115276
Pad2DStreamInt8<a,16,32,32,0,1,0,16> start = 6827,end = 122800,total = 115973
SplitInt8<a,16,1584,816,48>::filter2 start = 6872,end = 122838,total = 115966
QLinearConvHx4Stream<17,48,16,16,2,2,1,16,32,3> start = 122970,end = 169200,total = 46230
QLinearConvHx4Stream<17,48,16,16,2,2,1,16,32,3> start = 122974,end = 169204,total = 46230
ConcatInt8Stream<a,32,128,128,256> start = 6840,end = 169240,total = 162400
Pad2DStreamInt8<a,32,16,16,1,1,1,15> start = 6840,end = 169448,total = 162608
SplitInt8<a,32,576,320,64>::filter2 start = 6876,end = 169584,total = 162708
QLinearConvHx4Stream<10,32,16,16,1,1,1,32,32,3> start = 169717,end = 214142,total = 44425
QLinearConvHx4Stream<10,32,16,16,1,1,1,32,32,3> start = 169721,end = 214146,total = 44425
ConcatInt8Stream<a,32,128,128,256> start = 6844,end = 214181,total = 207337
QLinearConv1x1Stream<32,32,16,16,2,2,1,16,32,1> start = 122185,end = 217149,total = 94964
QLinearAddInt8<a,8192,0> start = 6916,end = 217192,total = 210276
Pad2DStreamInt8<a,32,16,16,0,1,0,16> start = 6849,end = 217272,total = 210423
SplitInt8<a,32,544,288,32>::filter2 start = 6888,end = 217331,total = 210443
QLinearConv1x1Stream<16,16,8,16,2,2,1,32,64,1> start = 217217,end = 287778,total = 70561
QLinearConvHx4StreamPad<9,32,8,16,2,2,1,32,64,3> start = 217461,end = 309743,total = 92282
QLinearConvHx4StreamPad<9,32,8,16,2,2,1,32,64,3> start = 217465,end = 309747,total = 92282
ConcatInt8Stream<a,64,64,64,128> start = 6856,end = 309764,total = 302908
Pad2DStreamInt8<a,64,8,16,1,1,1,15> start = 6865,end = 309902,total = 303037
SplitInt8<a,64,320,192,64>::filter2 start = 6904,end = 310016,total = 303112
QLinearAddInt8<a,4096,0> start = 6933,end = 365368,total = 358435
QLinearConvHx4StreamPad<6,32,8,16,1,1,1,64,64,3> start = 310146,end = 420406,total = 110260
QLinearConvHx4StreamPad<6,32,8,16,1,1,1,64,64,3> start = 310150,end = 420410,total = 110260
ConcatInt8Stream<a,64,64,64,128> start = 6872,end = 420427,total = 413555
QLinearAddInt8<a,4096,0> start = 365484,end = 420472,total = 54988
QLinearAvgpoolScalarBCHW<8,16,1,1,1,64> start = 420487,end = 437980,total = 17493
QgemmStream<1,64,16> start = 437981,end = 438104,total = 123
Running DequantizeLinear<1,16,12>
QlinearsoftmaxSingleaxis<1,10,16> start = 438117,end = 438288,total = 171
start = 438362,end = 438420,total = 58

Checking directories out: /home/ruien/workspace/onnx2versal/reports_dir/tiny_ic_int8/hw_emu/aiesimulator_output and data: /home/ruien/workspace/onnx2versal/data at tolerance @ rtol=1e-03, atol=1e-05
Checking 1/18: k0transpose_goldenout_shape1x3x32x32.txt                                         against k0transpose_goldenout_shape1x3x32x32.txt                TEST: OK!
Checking 2/18: k10qlinearconv_goldenout_shape1x64x8x8.txt                                       against k10qlinearconv_goldenout_shape1x64x8x8.txt              TEST: FAILED! Only 3959/4096 passed.
Max absolute difference: 1.0
Max relative difference: 0.047619047619047616
Checking 3/18: k11qlinearconv_goldenout_shape1x64x8x8.txt                                       against k11qlinearconv_goldenout_shape1x64x8x8.txt              TEST: FAILED! Only 3997/4096 passed.
Max absolute difference: 1.0
Max relative difference: 1.0
Checking 4/18: k12qlinearconv_goldenout_shape1x64x8x8.txt                                       against k12qlinearconv_goldenout_shape1x64x8x8.txt              TEST: FAILED! Only 3676/4096 passed.
Max absolute difference: 1.0
Max relative difference: 1.0
Checking 5/18: k13qlinearadd_goldenout_shape1x64x8x8.txt                                        against k13qlinearadd_goldenout_shape1x64x8x8.txt               TEST: FAILED! Only 3943/4096 passed.
Max absolute difference: 3.0
Max relative difference: 0.23076923076923078
Checking 6/18: k14qlinearpool_goldenout_shape1x64x1x1.txt                                       against k14qlinearpool_goldenout_shape1x64x1x1.txt              TEST: FAILED! Only 49/64 passed.
Max absolute difference: 1.0
Max relative difference: 0.041666666666666664
Checking 7/18: k16qgemm_goldenout_shape1x10.txt                                                 against k16qgemm_goldenout_shape1x10.txt                        TEST: FAILED! Only 7/10 passed.
Max absolute difference: 1.0
Max relative difference: 0.06666666666666667
Checking 8/18: k17qlinearsoftmax_goldenout_shape1x10.txt                                        against k17qlinearsoftmax_goldenout_shape1x10.txt               TEST: OK!
Checking 9/18: k18dequantizeLinear_goldenout_shape1x10.txt                                      against k18dequantizeLinear_goldenout_shape1x10.txt             TEST: OK!
Checking 10/18: k1quantizelinear_goldenout_shape1x3x32x32.txt                                    against k1quantizelinear_goldenout_shape1x3x32x32.txt          TEST: OK!
Checking 11/18: k2qlinearconv_goldenout_shape1x16x32x32.txt                                      against k2qlinearconv_goldenout_shape1x16x32x32.txt            TEST: OK!
Checking 12/18: k3qlinearconv_goldenout_shape1x16x32x32.txt                                      against k3qlinearconv_goldenout_shape1x16x32x32.txt            TEST: OK!
Checking 13/18: k4qlinearconv_goldenout_shape1x16x32x32.txt                                      against k4qlinearconv_goldenout_shape1x16x32x32.txt            TEST: FAILED! Only 16381/16384 passed.
Max absolute difference: 1.0
Max relative difference: 0.2
Checking 14/18: k5qlinearadd_goldenout_shape1x16x32x32.txt                                       against k5qlinearadd_goldenout_shape1x16x32x32.txt             TEST: FAILED! Only 16382/16384 passed.
Max absolute difference: 2.0
Max relative difference: 0.02702702702702703
Checking 15/18: k6qlinearconv_goldenout_shape1x32x16x16.txt                                      against k6qlinearconv_goldenout_shape1x32x16x16.txt            TEST: FAILED! Only 8178/8192 passed.
Max absolute difference: 1.0
Max relative difference: 0.011494252873563218
Checking 16/18: k7qlinearconv_goldenout_shape1x32x16x16.txt                                      against k7qlinearconv_goldenout_shape1x32x16x16.txt            TEST: FAILED! Only 8183/8192 passed.
Max absolute difference: 1.0
Max relative difference: 0.16666666666666666
Checking 17/18: k8qlinearconv_goldenout_shape1x32x16x16.txt                                      against k8qlinearconv_goldenout_shape1x32x16x16.txt            TEST: FAILED! Only 8105/8192 passed.
Max absolute difference: 1.0
Max relative difference: 1.0
Checking 18/18: k9qlinearadd_goldenout_shape1x32x16x16.txt                                       against k9qlinearadd_goldenout_shape1x32x16x16.txt             TEST: FAILED! Only 8145/8192 passed.
Max absolute difference: 3.0
Max relative difference: 0.045454545454545456
```

### tiny_kws
Note need ~26gb ram and ~2h for aiesimulator.
```
Waiting for core(s) of graph tiny_kws to finish execution ...
Pad2DStreamScalar<f,1,49,12,4,5,1,3> start = 8132,end = 9089,total = 957
ConvHx4ReluStream<58,16,5,8,2,2,1,1,64,10,4,1,1> start = 9106,end = 329160,total = 320054
Pad2DStreamScalar<f,64,25,8,1,1,1,3> start = 8137,end = 329177,total = 321040
SplitFilterFloatStreamTwice<f,64,324,84,24>::filter0 start = 8143,end = 329190,total = 321047
SplitFilterFloatStream<f,64,324,84,24>::filter4 start = 8136,end = 329195,total = 321059
SplitFilterFloatStreamTwice<f,64,324,84,24>::filter2 start = 8141,end = 329192,total = 321051
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 325579,end = 342740,total = 17161
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 326554,end = 343718,total = 17164
ConcatFloatStream<f,64,40,40,80> start = 8143,end = 343728,total = 335585
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 327531,end = 344692,total = 17161
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 328506,end = 345670,total = 17164
ConcatFloatStream<f,64,40,40,80> start = 8141,end = 345680,total = 337539
ConcatFloatStream<f,64,80,80,160> start = 8139,end = 345688,total = 337549
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 329203,end = 346364,total = 17161
ConcatFloatStream<f,64,160,40,200> start = 8134,end = 346373,total = 338239
SplitFilterFloatStreamTwice<f,64,200,40,0>::filter0 start = 8131,end = 346382,total = 338251
SplitFilterFloatStreamTwice<f,64,200,40,0>::filter2 start = 8129,end = 346384,total = 338255
SplitFilterFloatStream<f,64,200,40,0>::filter4 start = 8127,end = 346389,total = 338262
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 346188,end = 410608,total = 64420
ConcatFloatStream<f,64,40,40,80> start = 8131,end = 410656,total = 402525
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 346235,end = 410655,total = 64420
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 346278,end = 410698,total = 64420
ConcatFloatStream<f,64,40,40,80> start = 8129,end = 410746,total = 402617
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 346325,end = 410745,total = 64420
ConcatFloatStream<f,64,80,80,160> start = 8127,end = 410755,total = 402628
ConcatFloatStream<f,64,160,40,200> start = 8125,end = 410818,total = 402693
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 346398,end = 410818,total = 64420
Pad2DStreamScalar<f,64,25,8,1,1,1,3> start = 8120,end = 410890,total = 402770
SplitFilterFloatStream<f,64,324,84,24>::filter4 start = 8121,end = 410914,total = 402793
SplitFilterFloatStreamTwice<f,64,324,84,24>::filter0 start = 8119,end = 410918,total = 402799
SplitFilterFloatStreamTwice<f,64,324,84,24>::filter2 start = 8124,end = 410926,total = 402802
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 410677,end = 427838,total = 17161
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 410737,end = 427901,total = 17164
ConcatFloatStream<f,64,40,40,80> start = 8119,end = 427911,total = 419792
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 410805,end = 427966,total = 17161
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 410865,end = 428029,total = 17164
ConcatFloatStream<f,64,40,40,80> start = 8124,end = 428039,total = 419915
ConcatFloatStream<f,64,80,80,160> start = 8120,end = 428048,total = 419928
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 410923,end = 428084,total = 17161
ConcatFloatStream<f,64,160,40,200> start = 8115,end = 428106,total = 419991
SplitFilterFloatStreamTwice<f,64,200,40,0>::filter0 start = 8109,end = 428116,total = 420007
SplitFilterFloatStreamTwice<f,64,200,40,0>::filter2 start = 8113,end = 428120,total = 420007
SplitFilterFloatStream<f,64,200,40,0>::filter4 start = 8106,end = 428127,total = 420021
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 427922,end = 492342,total = 64420
ConcatFloatStream<f,64,40,40,80> start = 8109,end = 492390,total = 484281
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 427969,end = 492389,total = 64420
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 428022,end = 492442,total = 64420
ConcatFloatStream<f,64,40,40,80> start = 8113,end = 492502,total = 484389
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 428081,end = 492501,total = 64420
ConcatFloatStream<f,64,80,80,160> start = 8114,end = 492510,total = 484396
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 428138,end = 492558,total = 64420
ConcatFloatStream<f,64,160,40,200> start = 8112,end = 492565,total = 484453
Pad2DStreamScalar<f,64,25,8,1,1,1,3> start = 8109,end = 492623,total = 484514
SplitFilterFloatStream<f,64,324,84,24>::filter4 start = 8092,end = 492659,total = 484567
SplitFilterFloatStreamTwice<f,64,324,84,24>::filter0 start = 8084,end = 492663,total = 484579
SplitFilterFloatStreamTwice<f,64,324,84,24>::filter2 start = 8082,end = 492672,total = 484590
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 492422,end = 509583,total = 17161
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 492482,end = 509646,total = 17164
ConcatFloatStream<f,64,40,40,80> start = 8084,end = 509656,total = 501572
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 492551,end = 509712,total = 17161
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 492611,end = 509775,total = 17164
ConcatFloatStream<f,64,40,40,80> start = 8082,end = 509785,total = 501703
ConcatFloatStream<f,64,80,80,160> start = 8080,end = 509794,total = 501714
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 492668,end = 509829,total = 17161
ConcatFloatStream<f,64,160,40,200> start = 8079,end = 509849,total = 501770
SplitFilterFloatStreamTwice<f,64,200,40,0>::filter0 start = 8073,end = 509859,total = 501786
SplitFilterFloatStreamTwice<f,64,200,40,0>::filter2 start = 8079,end = 509865,total = 501786
SplitFilterFloatStream<f,64,200,40,0>::filter4 start = 8077,end = 509868,total = 501791
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 509665,end = 574085,total = 64420
ConcatFloatStream<f,64,40,40,80> start = 8073,end = 574133,total = 566060
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 509712,end = 574132,total = 64420
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 509767,end = 574187,total = 64420
ConcatFloatStream<f,64,40,40,80> start = 8079,end = 574247,total = 566168
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 509826,end = 574246,total = 64420
ConcatFloatStream<f,64,80,80,160> start = 8077,end = 574256,total = 566179
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 509877,end = 574297,total = 64420
ConcatFloatStream<f,64,160,40,200> start = 8075,end = 574312,total = 566237
Pad2DStreamScalar<f,64,25,8,1,1,1,3> start = 8080,end = 574368,total = 566288
SplitFilterFloatStream<f,64,324,84,24>::filter4 start = 8088,end = 574395,total = 566307
SplitFilterFloatStreamTwice<f,64,324,84,24>::filter0 start = 8093,end = 574401,total = 566308
SplitFilterFloatStreamTwice<f,64,324,84,24>::filter2 start = 8087,end = 574405,total = 566318
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 574160,end = 591321,total = 17161
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 574220,end = 591384,total = 17164
ConcatFloatStream<f,64,40,40,80> start = 8093,end = 591394,total = 583301
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 574284,end = 591445,total = 17161
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 574344,end = 591508,total = 17164
ConcatFloatStream<f,64,40,40,80> start = 8087,end = 591518,total = 583431
ConcatFloatStream<f,64,80,80,160> start = 8091,end = 591527,total = 583436
ConvHx4ReluStream<7,12,5,8,1,1,1,64,64,3,3,64,1> start = 574407,end = 591571,total = 17164
ConcatFloatStream<f,64,160,40,200> start = 8095,end = 591583,total = 583488
SplitFilterFloatStreamTwice<f,64,200,40,0>::filter0 start = 8097,end = 591594,total = 583497
SplitFilterFloatStreamTwice<f,64,200,40,0>::filter2 start = 8099,end = 591598,total = 583499
SplitFilterFloatStream<f,64,200,40,0>::filter4 start = 8103,end = 591605,total = 583502
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 591400,end = 655820,total = 64420
ConcatFloatStream<f,64,40,40,80> start = 8097,end = 655868,total = 647771
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 591447,end = 655867,total = 64420
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 591500,end = 655920,total = 64420
ConcatFloatStream<f,64,40,40,80> start = 8099,end = 655980,total = 647881
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 591559,end = 655979,total = 64420
ConcatFloatStream<f,64,80,80,160> start = 8101,end = 655991,total = 647890
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 591614,end = 656034,total = 64420
ConcatFloatStream<f,64,160,40,200> start = 8107,end = 656048,total = 647941
SplitScalar<f,1,12800,3200,0>::filter4 start = 8178,end = 780863,total = 772685
AvgpoolScalarBCHW<25,8,1,1,1,16,25,5> start = 780997,end = 785378,total = 4381
AvgpoolScalarBCHW<25,8,1,1,1,16,25,5> start = 781001,end = 785382,total = 4381
AvgpoolScalarBCHW<25,8,1,1,1,16,25,5> start = 781005,end = 785386,total = 4381
AvgpoolScalarBCHW<25,8,1,1,1,16,25,5> start = 781009,end = 785390,total = 4381
ConcatFloatStream<f,1,16,16,32> start = 8105,end = 785548,total = 777443
ConcatFloatStream<f,1,16,16,32> start = 8106,end = 785556,total = 777450
ConcatFloatStream<f,1,32,32,64> start = 8101,end = 785598,total = 777497
GemmReluMKKN<1,64,12,0> start = 785613,end = 785880,total = 267
ConcatFloat<f,1,1,12,12>::filter1 start = 786004,end = 786044,total = 40
SoftmaxScalar<1,12,12> start = 786063,end = 823206,total = 37143
core(s) are done executing
[INFO]: Fifo Guidance report generated ./AIESim_FIFO_Guidance.json
Exiting!
generate profile data for all cores
generate profile data for all cores
Stopping Simulator.

Info: /OSCI/SystemC: Simulation stopped by user.
IP-INFO: deleting ip PSIP_ps_i135 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
[INFO] : Simulation Finished, Sim result: 0
Checking directories out: /home/ruien/workspace/onnx2versal/reports_dir/tiny_kws/hw_emu/aiesimulator_output and data: /home/ruien/workspace/onnx2versal/data at tolerance @ rtol=1e-03, atol=1e-05
Checking 1/12: k11conv_goldenout_shape1x64x25x5.txt                                             against k11conv_goldenout_shape1x64x25x5.txt                    TEST: OK!
Checking 2/12: k13conv_goldenout_shape1x64x25x5.txt                                             against k13conv_goldenout_shape1x64x25x5.txt                    TEST: OK!
Checking 3/12: k15conv_goldenout_shape1x64x25x5.txt                                             against k15conv_goldenout_shape1x64x25x5.txt                    TEST: OK!
Checking 4/12: k17conv_goldenout_shape1x64x25x5.txt                                             against k17conv_goldenout_shape1x64x25x5.txt                    TEST: FAILED! Only 7994/8000 passed.
Max absolute difference: 6.198900000065066e-05
Max relative difference: 0.17475247609781588
Checking 5/12: k19pool_goldenout_shape1x64x1x1.txt                                              against k19pool_goldenout_shape1x64x1x1.txt                     TEST: OK!
Checking 6/12: k1conv_goldenout_shape1x64x25x5.txt                                              against k1conv_goldenout_shape1x64x25x5.txt                     TEST: OK!
Checking 7/12: k21gemm_goldenout_shape1x12.txt                                                  against k21gemm_goldenout_shape1x12.txt                         TEST: OK!
Checking 8/12: k23softmax_goldenout_shape1x12.txt                                               against k23softmax_goldenout_shape1x12.txt                      TEST: OK!
Checking 9/12: k3conv_goldenout_shape1x64x25x5.txt                                              against k3conv_goldenout_shape1x64x25x5.txt                     TEST: OK!
Checking 10/12: k5conv_goldenout_shape1x64x25x5.txt                                              against k5conv_goldenout_shape1x64x25x5.txt                    TEST: OK!
Checking 11/12: k7conv_goldenout_shape1x64x25x5.txt                                              against k7conv_goldenout_shape1x64x25x5.txt                    TEST: OK!
Checking 12/12: k9conv_goldenout_shape1x64x25x5.txt                                              against k9conv_goldenout_shape1x64x25x5.txt                    TEST: OK!
Traceback (most recent call last):
  File "/home/ruien/workspace/onnx2versal/check.py", line 98, in <module>
    assert(pass_count == len(filepairs)), f"{pass_count} / {len(filepairs)} tests passed."
AssertionError: 11 / 12 tests passed.
```

### tiny_kws_int8
Note need ~26gb ram and ~1h for aiesimulator.
```
QuantizeLinearFmulStream<49,12,16> start = 5725,end = 7290,total = 1565
Pad2DStreamInt8<a,1,49,16,4,5,1,15> start = 5726,end = 7381,total = 1655
QLinearConvHx4StreamPad<58,32,5,16,2,2,1,1,64,10,4,1> start = 7403,end = 111787,total = 104384
Pad2DStreamInt8<a,64,25,16,1,1,1,15> start = 5729,end = 111851,total = 106122
SplitFilterInt8StreamTwice<a,64,864,224,64>::filter2 start = 5732,end = 111867,total = 106135
SplitFilterInt8Stream<a,64,864,224,64>::filter4 start = 5735,end = 111868,total = 106133
SplitFilterInt8StreamTwice<a,64,864,224,64>::filter0 start = 5738,end = 111870,total = 106132
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 110756,end = 119075,total = 8319
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 111070,end = 119389,total = 8319
ConcatInt8Stream<a,64,80,80,160> start = 5738,end = 119400,total = 113662
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 111383,end = 119702,total = 8319
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 111697,end = 120016,total = 8319
ConcatInt8Stream<a,64,80,80,160> start = 5732,end = 120027,total = 114295
ConcatInt8Stream<a,64,160,160,320> start = 5736,end = 120038,total = 114302
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 111886,end = 120205,total = 8319
ConcatInt8Stream<a,64,320,80,400> start = 5735,end = 120217,total = 114482
SplitFilterInt8Stream<a,64,400,80,0>::filter4 start = 5722,end = 120237,total = 114515
SplitFilterInt8StreamTwice<a,64,400,80,0>::filter2 start = 5725,end = 120237,total = 114512
SplitFilterInt8StreamTwice<a,64,400,80,0>::filter0 start = 5721,end = 120239,total = 114518
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 120160,end = 159560,total = 39400
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 120179,end = 159579,total = 39400
ConcatInt8Stream<a,64,80,80,160> start = 5721,end = 159588,total = 153867
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 120203,end = 159603,total = 39400
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 120222,end = 159622,total = 39400
ConcatInt8Stream<a,64,80,80,160> start = 5725,end = 159631,total = 153906
ConcatInt8Stream<a,64,160,160,320> start = 5723,end = 159641,total = 153918
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 120253,end = 159653,total = 39400
ConcatInt8Stream<a,64,320,80,400> start = 5720,end = 159673,total = 153953
Pad2DStreamInt8<a,64,25,16,1,1,1,15> start = 5711,end = 160108,total = 154397
SplitFilterInt8Stream<a,64,864,224,64>::filter4 start = 5709,end = 160123,total = 154414
SplitFilterInt8StreamTwice<a,64,864,224,64>::filter0 start = 5714,end = 160124,total = 154410
SplitFilterInt8StreamTwice<a,64,864,224,64>::filter2 start = 5711,end = 160125,total = 154414
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 159690,end = 168009,total = 8319
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 159804,end = 168123,total = 8319
ConcatInt8Stream<a,64,80,80,160> start = 5714,end = 168134,total = 162420
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 159921,end = 168240,total = 8319
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 160035,end = 168354,total = 8319
ConcatInt8Stream<a,64,80,80,160> start = 5711,end = 168365,total = 162654
ConcatInt8Stream<a,64,160,160,320> start = 5709,end = 168376,total = 162667
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 160139,end = 168458,total = 8319
ConcatInt8Stream<a,64,320,80,400> start = 5706,end = 168468,total = 162762
SplitFilterInt8StreamTwice<a,64,400,80,0>::filter2 start = 5704,end = 168484,total = 162780
SplitFilterInt8StreamTwice<a,64,400,80,0>::filter0 start = 5700,end = 168486,total = 162786
SplitFilterInt8Stream<a,64,400,80,0>::filter4 start = 5695,end = 168487,total = 162792
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 168407,end = 207807,total = 39400
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 168426,end = 207826,total = 39400
ConcatInt8Stream<a,64,80,80,160> start = 5700,end = 207835,total = 202135
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 168450,end = 207850,total = 39400
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 168469,end = 207869,total = 39400
ConcatInt8Stream<a,64,80,80,160> start = 5704,end = 207878,total = 202174
ConcatInt8Stream<a,64,160,160,320> start = 5702,end = 207888,total = 202186
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 168503,end = 207903,total = 39400
ConcatInt8Stream<a,64,320,80,400> start = 5698,end = 207919,total = 202221
Pad2DStreamInt8<a,64,25,16,1,1,1,15> start = 5694,end = 208352,total = 202658
SplitFilterInt8Stream<a,64,864,224,64>::filter4 start = 5693,end = 208367,total = 202674
SplitFilterInt8StreamTwice<a,64,864,224,64>::filter2 start = 5693,end = 208368,total = 202675
SplitFilterInt8StreamTwice<a,64,864,224,64>::filter0 start = 5696,end = 208368,total = 202672
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 207934,end = 216253,total = 8319
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 208048,end = 216367,total = 8319
ConcatInt8Stream<a,64,80,80,160> start = 5696,end = 216378,total = 210682
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 208164,end = 216483,total = 8319
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 208278,end = 216597,total = 8319
ConcatInt8Stream<a,64,80,80,160> start = 5693,end = 216608,total = 210915
ConcatInt8Stream<a,64,160,160,320> start = 5694,end = 216618,total = 210924
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 208383,end = 216702,total = 8319
ConcatInt8Stream<a,64,320,80,400> start = 5686,end = 216714,total = 211028
SplitFilterInt8Stream<a,64,400,80,0>::filter4 start = 5683,end = 216729,total = 211046
SplitFilterInt8StreamTwice<a,64,400,80,0>::filter2 start = 5688,end = 216730,total = 211042
SplitFilterInt8StreamTwice<a,64,400,80,0>::filter0 start = 5686,end = 216732,total = 211046
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 216653,end = 256053,total = 39400
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 216672,end = 256072,total = 39400
ConcatInt8Stream<a,64,80,80,160> start = 5686,end = 256081,total = 250395
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 216696,end = 256096,total = 39400
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 216715,end = 256115,total = 39400
ConcatInt8Stream<a,64,80,80,160> start = 5688,end = 256124,total = 250436
ConcatInt8Stream<a,64,160,160,320> start = 5684,end = 256135,total = 250451
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 216745,end = 256145,total = 39400
ConcatInt8Stream<a,64,320,80,400> start = 5683,end = 256165,total = 250482
Pad2DStreamInt8<a,64,25,16,1,1,1,15> start = 5681,end = 256597,total = 250916
SplitFilterInt8StreamTwice<a,64,864,224,64>::filter0 start = 5680,end = 256613,total = 250933
SplitFilterInt8Stream<a,64,864,224,64>::filter4 start = 5673,end = 256615,total = 250942
SplitFilterInt8StreamTwice<a,64,864,224,64>::filter2 start = 5677,end = 256616,total = 250939
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 256179,end = 264498,total = 8319
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 256293,end = 264612,total = 8319
ConcatInt8Stream<a,64,80,80,160> start = 5680,end = 264623,total = 258943
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 256412,end = 264731,total = 8319
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 256526,end = 264845,total = 8319
ConcatInt8Stream<a,64,80,80,160> start = 5677,end = 264856,total = 259179
ConcatInt8Stream<a,64,160,160,320> start = 5681,end = 264867,total = 259186
QLinearConvHx4StreamPad<7,32,5,16,1,1,1,64,64,3,3,64> start = 256631,end = 264950,total = 8319
ConcatInt8Stream<a,64,320,80,400> start = 5675,end = 264961,total = 259286
SplitFilterInt8Stream<a,64,400,80,0>::filter4 start = 5668,end = 264978,total = 259310
SplitFilterInt8StreamTwice<a,64,400,80,0>::filter2 start = 5671,end = 264978,total = 259307
SplitFilterInt8StreamTwice<a,64,400,80,0>::filter0 start = 5673,end = 264980,total = 259307
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 264901,end = 304301,total = 39400
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 264920,end = 304320,total = 39400
ConcatInt8Stream<a,64,80,80,160> start = 5673,end = 304329,total = 298656
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 264944,end = 304344,total = 39400
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 264963,end = 304363,total = 39400
ConcatInt8Stream<a,64,80,80,160> start = 5671,end = 304372,total = 298701
ConcatInt8Stream<a,64,160,160,320> start = 5675,end = 304384,total = 298709
QLinearConv1x1Stream<5,16,5,16,1,1,1,64,64,1,1,1> start = 264994,end = 304394,total = 39400
ConcatInt8Stream<a,64,320,80,400> start = 5670,end = 304418,total = 298748
SplitInt8<a,1,25600,12800,0>::filter2 start = 5708,end = 304652,total = 298944
ConcatInt8Stream<a,1,32,32,64> start = 5675,end = 317462,total = 311787
QgemmStream<1,64,16> start = 317474,end = 317590,total = 116
QlinearsoftmaxSingleaxis<1,12,16> start = 317602,end = 317773,total = 171
DequantizeLinear<1,16,12> start = 317792,end = 317858,total = 66
QLinearAvgpoolScalarBCHW<25,16,1,1,1,64,25,5> start = 304782,end = 330083,total = 25301
QLinearAvgpoolScalarBCHW<25,16,1,1,1,64,25,5> start = 304786,end = 330087,total = 25301
core(s) are done executing
[INFO]: Fifo Guidance report generated ./AIESim_FIFO_Guidance.json
Exiting!
generate profile data for all cores
generate profile data for all cores
Stopping Simulator.

Info: /OSCI/SystemC: Simulation stopped by user.
IP-INFO: deleting ip PSIP_ps_i134 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
IP-INFO: deleting packet ip 
[INFO] : Simulation Finished, Sim result: 0
Checking directories out: /home/ruien/workspace/onnx2versal/reports_dir/tiny_kws_int8/hw_emu/aiesimulator_output and data: /home/ruien/workspace/onnx2versal/data at tolerance @ rtol=1e-03, atol=1e-05
Checking 1/14: k10qlinearconv_goldenout_shape1x64x25x5.txt                                      against k10qlinearconv_goldenout_shape1x64x25x5.txt             TEST: FAILED! Only 7972/8000 passed.
Max absolute difference: 1.0
Max relative difference: 0.017857142857142856
Checking 2/14: k11qlinearpool_goldenout_shape1x64x1x1.txt                                       against k11qlinearpool_goldenout_shape1x64x1x1.txt              TEST: FAILED! Only 63/64 passed.
Max absolute difference: 1.0
Max relative difference: 0.017241379310344827
Checking 3/14: k13qgemm_goldenout_shape1x12.txt                                                 against k13qgemm_goldenout_shape1x12.txt                        TEST: OK!
Checking 4/14: k14qlinearsoftmax_goldenout_shape1x12.txt                                        against k14qlinearsoftmax_goldenout_shape1x12.txt               TEST: OK!
Checking 5/14: k15dequantizeLinear_goldenout_shape1x12.txt                                      against k15dequantizeLinear_goldenout_shape1x12.txt             TEST: OK!
Checking 6/14: k1quantizelinear_goldenout_shape1x1x49x10.txt                                    against k1quantizelinear_goldenout_shape1x1x49x10.txt           TEST: OK!
Checking 7/14: k2qlinearconv_goldenout_shape1x64x25x5.txt                                       against k2qlinearconv_goldenout_shape1x64x25x5.txt              TEST: OK!
Checking 8/14: k3qlinearconv_goldenout_shape1x64x25x5.txt                                       against k3qlinearconv_goldenout_shape1x64x25x5.txt              TEST: OK!
Checking 9/14: k4qlinearconv_goldenout_shape1x64x25x5.txt                                       against k4qlinearconv_goldenout_shape1x64x25x5.txt              TEST: OK!
Checking 10/14: k5qlinearconv_goldenout_shape1x64x25x5.txt                                       against k5qlinearconv_goldenout_shape1x64x25x5.txt             TEST: OK!
Checking 11/14: k6qlinearconv_goldenout_shape1x64x25x5.txt                                       against k6qlinearconv_goldenout_shape1x64x25x5.txt             TEST: OK!
Checking 12/14: k7qlinearconv_goldenout_shape1x64x25x5.txt                                       against k7qlinearconv_goldenout_shape1x64x25x5.txt             TEST: FAILED! Only 7999/8000 passed.
Max absolute difference: 1.0
Max relative difference: 0.010309278350515464
Checking 13/14: k8qlinearconv_goldenout_shape1x64x25x5.txt                                       against k8qlinearconv_goldenout_shape1x64x25x5.txt             TEST: FAILED! Only 7991/8000 passed.
Max absolute difference: 1.0
Max relative difference: 0.024390243902439025
Checking 14/14: k9qlinearconv_goldenout_shape1x64x25x5.txt                                       against k9qlinearconv_goldenout_shape1x64x25x5.txt             TEST: FAILED! Only 7996/8000 passed.
Max absolute difference: 1.0
Max relative difference: 0.019230769230769232
```

### tiny_vww
```
```

### tiny_vww_int8
```
```