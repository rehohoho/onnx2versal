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
GemmReluMKKNStream<14,640,128,0> start = 169041,end = 743098,total = 574057
MacFloat<f,14,128,1> start = 745036,end = 745998,total = 962
GemmReluMKKN<14,128,32,0> start = 747933,end = 755710,total = 7777
GemmReluMKKN<14,128,32,0> start = 747934,end = 755711,total = 7777
GemmReluMKKN<14,128,32,0> start = 747936,end = 755713,total = 7777
GemmReluMKKN<14,128,32,0> start = 747938,end = 755715,total = 7777
ConcatFloatStream<f,14,32,32,64> start = 169036,end = 757019,total = 587983
ConcatFloatStream<f,14,32,32,64> start = 169040,end = 757021,total = 587981
ConcatFloatStream<f,14,64,64,128> start = 169038,end = 757932,total = 588894
MacFloat<f,14,128,1> start = 757944,end = 758906,total = 962
GemmReluMKKN<14,128,32,0> start = 760846,end = 768623,total = 7777
GemmReluMKKN<14,128,32,0> start = 760849,end = 768626,total = 7777
GemmReluMKKN<14,128,32,0> start = 760851,end = 768628,total = 7777
GemmReluMKKN<14,128,32,0> start = 760851,end = 768628,total = 7777
ConcatFloatStream<f,14,32,32,64> start = 169032,end = 769935,total = 600903
ConcatFloatStream<f,14,32,32,64> start = 169028,end = 769937,total = 600909
ConcatFloatStream<f,14,64,64,128> start = 169030,end = 770848,total = 601818
MacFloat<f,14,128,1> start = 770861,end = 771823,total = 962
GemmReluMKKN<14,128,32,0> start = 773759,end = 781536,total = 7777
GemmReluMKKN<14,128,32,0> start = 773761,end = 781538,total = 7777
GemmReluMKKN<14,128,32,0> start = 773762,end = 781539,total = 7777
GemmReluMKKN<14,128,32,0> start = 773763,end = 781540,total = 7777
ConcatFloatStream<f,14,32,32,64> start = 169025,end = 782843,total = 613818
ConcatFloatStream<f,14,32,32,64> start = 169023,end = 782850,total = 613827
ConcatFloatStream<f,14,64,64,128> start = 169022,end = 783755,total = 614733
MacFloat<f,14,128,1> start = 783766,end = 784728,total = 962
GemmReluMKKN<14,128,8,0> start = 786664,end = 792083,total = 5419
MacFloat<f,14,8,1> start = 792337,end = 792498,total = 161
GemmReluMKKN<14,8,128,0> start = 792761,end = 797471,total = 4710
MacFloat<f,14,128,1> start = 799405,end = 800367,total = 962
GemmReluMKKN<14,128,32,0> start = 802307,end = 810084,total = 7777
GemmReluMKKN<14,128,32,0> start = 802309,end = 810086,total = 7777
GemmReluMKKN<14,128,32,0> start = 802310,end = 810087,total = 7777
GemmReluMKKN<14,128,32,0> start = 802312,end = 810089,total = 7777
ConcatFloatStream<f,14,32,32,64> start = 169020,end = 811390,total = 642370
ConcatFloatStream<f,14,32,32,64> start = 169021,end = 811393,total = 642372
ConcatFloatStream<f,14,64,64,128> start = 169018,end = 812305,total = 643287
MacFloat<f,14,128,1> start = 812318,end = 813280,total = 962
GemmReluMKKN<14,128,32,0> start = 815216,end = 822993,total = 7777
GemmReluMKKN<14,128,32,0> start = 815219,end = 822996,total = 7777
GemmReluMKKN<14,128,32,0> start = 815221,end = 822998,total = 7777
GemmReluMKKN<14,128,32,0> start = 815224,end = 823001,total = 7777
ConcatFloatStream<f,14,32,32,64> start = 169015,end = 824304,total = 655289
ConcatFloatStream<f,14,32,32,64> start = 169016,end = 824310,total = 655294
ConcatFloatStream<f,14,64,64,128> start = 169014,end = 825215,total = 656201
MacFloat<f,14,128,1> start = 825228,end = 826190,total = 962
GemmReluMKKN<14,128,32,0> start = 828126,end = 835903,total = 7777
GemmReluMKKN<14,128,32,0> start = 828130,end = 835907,total = 7777
GemmReluMKKN<14,128,32,0> start = 828132,end = 835909,total = 7777
GemmReluMKKN<14,128,32,0> start = 828133,end = 835910,total = 7777
ConcatFloatStream<f,14,32,32,64> start = 169008,end = 837217,total = 668209
ConcatFloatStream<f,14,32,32,64> start = 169008,end = 837218,total = 668210
ConcatFloatStream<f,14,64,64,128> start = 169006,end = 838129,total = 669123
MacFloat<f,14,128,1> start = 838141,end = 839103,total = 962
GemmReluMKKNStream<2,128,640,0> start = 169025,end = 1007055,total = 838030
GemmReluMKKNStream<2,128,640,0> start = 1007198,end = 1175045,total = 167847
GemmReluMKKNStream<2,128,640,0> start = 1175188,end = 1343035,total = 167847
GemmReluMKKNStream<2,128,640,0> start = 1343178,end = 1511025,total = 167847
GemmReluMKKNStream<2,128,640,0> start = 1511168,end = 1679015,total = 167847
GemmReluMKKNStream<2,128,640,0> start = 1679158,end = 1847005,total = 167847
GemmReluMKKNStream<2,128,640,0> start = 1847148,end = 2014995,total = 167847

Checking 1/19: k000gemm_goldenout_shape14x128.txt                                               against k000gemm_goldenout_shape14x128.txt                      TEST: OK!
Checking 2/19: k002mac_goldenout_shape14x128.txt                                                against k002mac_goldenout_shape14x128.txt                       TEST: OK!
Checking 3/19: k005gemm_goldenout_shape14x128.txt                                               against k005gemm_goldenout_shape14x128.txt                      TEST: FAILED! Only 1785/1792 passed.
Max absolute difference: 0.0002851489999997625
Max relative difference: 0.007601085531505405
Checking 4/19: k007mac_goldenout_shape14x128.txt                                                against k007mac_goldenout_shape14x128.txt                       TEST: OK!
Checking 5/19: k010gemm_goldenout_shape14x128.txt                                               against k010gemm_goldenout_shape14x128.txt                      TEST: FAILED! Only 1790/1792 passed.
Max absolute difference: 0.00018501000000092915
Max relative difference: 0.004238935443148665
Checking 6/19: k012mac_goldenout_shape14x128.txt                                                against k012mac_goldenout_shape14x128.txt                       TEST: OK!
Checking 7/19: k015gemm_goldenout_shape14x128.txt                                               against k015gemm_goldenout_shape14x128.txt                      TEST: FAILED! Only 1791/1792 passed.
Max absolute difference: 0.0001485350000001162
Max relative difference: 0.007625754539909896
Checking 8/19: k017mac_goldenout_shape14x128.txt                                                against k017mac_goldenout_shape14x128.txt                       TEST: OK!
Checking 9/19: k020gemm_goldenout_shape14x8.txt                                                 against k020gemm_goldenout_shape14x8.txt                        TEST: OK!
Checking 10/19: k022mac_goldenout_shape14x8.txt                                                  against k022mac_goldenout_shape14x8.txt                        TEST: OK!
Checking 11/19: k025gemm_goldenout_shape14x128.txt                                               against k025gemm_goldenout_shape14x128.txt                     TEST: OK!
Checking 12/19: k027mac_goldenout_shape14x128.txt                                                against k027mac_goldenout_shape14x128.txt                      TEST: FAILED! Only 1791/1792 passed.
Max absolute difference: 8.869100000019614e-05
Max relative difference: 0.003061849624789399
Checking 13/19: k030gemm_goldenout_shape14x128.txt                                               against k030gemm_goldenout_shape14x128.txt                     TEST: FAILED! Only 1783/1792 passed.
Max absolute difference: 0.0002965929999998451
Max relative difference: 0.0036314493565588355
Checking 14/19: k032mac_goldenout_shape14x128.txt                                                against k032mac_goldenout_shape14x128.txt                      TEST: OK!
Checking 15/19: k035gemm_goldenout_shape14x128.txt                                               against k035gemm_goldenout_shape14x128.txt                     TEST: FAILED! Only 1776/1792 passed.
Max absolute difference: 0.0003018380000003873
Max relative difference: 0.1699281504671081
Checking 16/19: k037mac_goldenout_shape14x128.txt                                                against k037mac_goldenout_shape14x128.txt                      TEST: FAILED! Only 1790/1792 passed.
Max absolute difference: 7.48630000000361e-05
Max relative difference: 0.004239780000101313
Checking 17/19: k040gemm_goldenout_shape14x128.txt                                               against k040gemm_goldenout_shape14x128.txt                     TEST: FAILED! Only 1775/1792 passed.
Max absolute difference: 0.00018620499999988382
Max relative difference: 0.01272612987484207
Checking 18/19: k042mac_goldenout_shape14x128.txt                                                against k042mac_goldenout_shape14x128.txt                      TEST: OK!
Checking 19/19: k045gemm_goldenout_shape14x640.txt                                               against k045gemm_goldenout_shape14x640.txt                     TEST: FAILED! Only 8956/8960 passed.
Max absolute difference: 0.000354770000001281
Max relative difference: 0.01695552934752426
```

### tiny_ad_int8

```
DequantizeLinear<a,2,640,640> start = 18189,end = 19799,total = 1610
QuantizeLinearFmulStream<a,14,640,640> start = 2093,end = 20074,total = 17981
QgemmStream<a,a,14,640,16> start = 2214,end = 20125,total = 17911
QgemmStream<a,a,14,640,16> start = 2218,end = 20125,total = 17907
QgemmStream<a,a,14,640,16> start = 2213,end = 20126,total = 17913
QgemmStream<a,a,14,640,16> start = 2217,end = 20126,total = 17909
QgemmStream<a,a,14,640,16> start = 2220,end = 20127,total = 17907
QgemmStream<a,a,14,640,16> start = 2221,end = 20128,total = 17907
QgemmStream<a,a,14,640,16> start = 2216,end = 20129,total = 17913
QgemmStream<a,a,14,640,16> start = 2217,end = 20130,total = 17913
ConcatInt8Stream<a,14,16,16,32> start = 2093,end = 20141,total = 18048
ConcatInt8Stream<a,14,16,16,32> start = 2097,end = 20141,total = 18044
ConcatInt8Stream<a,14,16,16,32> start = 2100,end = 20142,total = 18042
ConcatInt8Stream<a,14,16,16,32> start = 2096,end = 20144,total = 18048
ConcatInt8Stream<a,14,32,32,64> start = 2095,end = 20159,total = 18064
ConcatInt8Stream<a,14,32,32,64> start = 2099,end = 20159,total = 18060
ConcatInt8Stream<a,14,64,64,128> start = 2093,end = 20188,total = 18095
QlinearMac<a,a,14,128,0> start = 5373,end = 20373,total = 15000
QgemmStream<a,a,14,128,128> start = 2207,end = 22341,total = 20134
DequantizeLinear<a,2,640,640> start = 20756,end = 22366,total = 1610
QlinearMac<a,a,14,128,0> start = 5371,end = 22389,total = 17018
QgemmStream<a,a,14,128,128> start = 2210,end = 23556,total = 21346
QlinearMac<a,a,14,128,0> start = 5382,end = 23608,total = 18226
QgemmStream<a,a,14,128,128> start = 2219,end = 24773,total = 22554
DequantizeLinear<a,2,640,640> start = 23168,end = 24778,total = 1610
QlinearMac<a,a,14,128,0> start = 5383,end = 24821,total = 19438
QgemmStream<a,a,14,128,16> start = 2222,end = 24872,total = 22650
QlinearMac<a,a,14,16,0> start = 2585,end = 24917,total = 22332
QgemmStream<a,a,14,16,128> start = 2225,end = 25490,total = 23265
QlinearMac<a,a,14,128,0> start = 5403,end = 25543,total = 20140
DequantizeLinear<a,2,640,640> start = 25444,end = 27054,total = 1610
QgemmStream<a,a,14,128,128> start = 2234,end = 28433,total = 26199
QlinearMac<a,a,14,128,0> start = 5402,end = 28480,total = 23078
DequantizeLinear<a,2,640,640> start = 27727,end = 29337,total = 1610
QgemmStream<a,a,14,128,128> start = 2237,end = 29656,total = 27419
QlinearMac<a,a,14,128,0> start = 5401,end = 29704,total = 24303
QgemmStream<a,a,14,128,128> start = 2240,end = 30870,total = 28630
QlinearMac<a,a,14,128,0> start = 5398,end = 30919,total = 25521
DequantizeLinear<a,2,640,640> start = 29999,end = 31609,total = 1610
QgemmStream<a,a,14,128,128> start = 2234,end = 32088,total = 29854
QgemmStream<a,a,14,128,128> start = 2235,end = 32089,total = 29854
QgemmStream<a,a,14,128,128> start = 2226,end = 32091,total = 29865
QgemmStream<a,a,14,128,128> start = 2231,end = 32092,total = 29861
QgemmStream<a,a,14,128,128> start = 2232,end = 32099,total = 29867
ConcatInt8Stream<a,14,128,128,256> start = 2114,end = 32131,total = 30017
ConcatInt8Stream<a,14,128,128,256> start = 2111,end = 32135,total = 30024
ConcatInt8Stream<a,14,256,256,512> start = 2107,end = 32210,total = 30103
ConcatInt8Stream<a,14,512,128,640> start = 2103,end = 32253,total = 30150
DequantizeLinear<a,2,640,640> start = 32272,end = 33882,total = 1610


Checking 1/21: k000quantizelinear_goldenout_shape14x640.txt                                     against k000quantizelinear_goldenout_shape14x640.txt            TEST: OK!
Checking 2/21: k001qgemm_goldenout_shape14x128.txt                                              against k001qgemm_goldenout_shape14x128.txt                     TEST: OK!
Checking 3/21: k002qlinearmac_goldenout_shape14x128.txt                                         against k002qlinearmac_goldenout_shape14x128.txt                TEST: FAILED! Only 1789/1792 passed.
Max absolute difference: 1.0
Max relative difference: 0.008
Checking 4/21: k004qgemm_goldenout_shape14x128.txt                                              against k004qgemm_goldenout_shape14x128.txt                     TEST: FAILED! Only 1779/1792 passed.
Max absolute difference: 1.0
Max relative difference: 0.025
Checking 5/21: k005qlinearmac_goldenout_shape14x128.txt                                         against k005qlinearmac_goldenout_shape14x128.txt                TEST: FAILED! Only 1786/1792 passed.
Max absolute difference: 7.0
Max relative difference: 0.06930693069306931
Checking 6/21: k007qgemm_goldenout_shape14x128.txt                                              against k007qgemm_goldenout_shape14x128.txt                     TEST: FAILED! Only 1703/1792 passed.
Max absolute difference: 5.0
Max relative difference: 0.09615384615384616
Checking 7/21: k008qlinearmac_goldenout_shape14x128.txt                                         against k008qlinearmac_goldenout_shape14x128.txt                TEST: FAILED! Only 1753/1792 passed.
Max absolute difference: 25.0
Max relative difference: 0.22321428571428573
Checking 8/21: k010qgemm_goldenout_shape14x128.txt                                              against k010qgemm_goldenout_shape14x128.txt                     TEST: FAILED! Only 1531/1792 passed.
Max absolute difference: 9.0
Max relative difference: 6.0
Checking 9/21: k011qlinearmac_goldenout_shape14x128.txt                                         against k011qlinearmac_goldenout_shape14x128.txt                TEST: FAILED! Only 1711/1792 passed.
Max absolute difference: 12.0
Max relative difference: 0.10434782608695652
Checking 10/21: k013qgemm_goldenout_shape14x8.txt                                                against k013qgemm_goldenout_shape14x8.txt                      TEST: FAILED! Only 91/112 passed.
Max absolute difference: 3.0
Max relative difference: 2.0
Checking 11/21: k014qlinearmac_goldenout_shape14x8.txt                                           against k014qlinearmac_goldenout_shape14x8.txt                 TEST: FAILED! Only 91/112 passed.
Max absolute difference: 5.0
Max relative difference: 1.0
Checking 12/21: k016qgemm_goldenout_shape14x128.txt                                              against k016qgemm_goldenout_shape14x128.txt                    TEST: FAILED! Only 1578/1792 passed.
Max absolute difference: 2.0
Max relative difference: 2.0
Checking 13/21: k017qlinearmac_goldenout_shape14x128.txt                                         against k017qlinearmac_goldenout_shape14x128.txt               TEST: FAILED! Only 1700/1792 passed.
Max absolute difference: 11.0
Max relative difference: 0.1346153846153846
Checking 14/21: k019qgemm_goldenout_shape14x128.txt                                              against k019qgemm_goldenout_shape14x128.txt                    TEST: FAILED! Only 1325/1792 passed.
Max absolute difference: 7.0
Max relative difference: 4.0
Checking 15/21: k020qlinearmac_goldenout_shape14x128.txt                                         against k020qlinearmac_goldenout_shape14x128.txt               TEST: FAILED! Only 1582/1792 passed.
Max absolute difference: 14.0
Max relative difference: 0.23728813559322035
Checking 16/21: k022qgemm_goldenout_shape14x128.txt                                              against k022qgemm_goldenout_shape14x128.txt                    TEST: FAILED! Only 1178/1792 passed.
Max absolute difference: 8.0
Max relative difference: 3.0
Checking 17/21: k023qlinearmac_goldenout_shape14x128.txt                                         against k023qlinearmac_goldenout_shape14x128.txt               TEST: FAILED! Only 1487/1792 passed.
Max absolute difference: 16.0
Max relative difference: 0.32
Checking 18/21: k025qgemm_goldenout_shape14x128.txt                                              against k025qgemm_goldenout_shape14x128.txt                    TEST: FAILED! Only 1105/1792 passed.
Max absolute difference: 8.0
Max relative difference: 8.0
Checking 19/21: k026qlinearmac_goldenout_shape14x128.txt                                         against k026qlinearmac_goldenout_shape14x128.txt               TEST: FAILED! Only 1260/1792 passed.
Max absolute difference: 20.0
Max relative difference: 1.0
Checking 20/21: k028qgemm_goldenout_shape14x640.txt                                              against k028qgemm_goldenout_shape14x640.txt                    TEST: FAILED! Only 6080/8960 passed.
Max absolute difference: 10.0
Max relative difference: 3.0
Checking 21/21: k029dequantizeLinear_goldenout_shape14x640.txt                                   against k029dequantizeLinear_goldenout_shape14x640.txt         TEST: FAILED! Only 6074/8960 passed.
Max absolute difference: 3.751939773
Max relative difference: 2.9998281126529553
```

### tiny_ic
```
TransposeScalarBHWC2BCHW<f,1,32,32,3> start = 17894,end = 24289,total = 6395
Pad2DStreamFloat<f,3,32,32,32,1,1,1,3> start = 17863,end = 28117,total = 10254
ConvHx4ReluStream<34,36,32,32,1,1,1,3,16,3,3,1,1> start = 28135,end = 170187,total = 142052
Pad2DStreamFloat<f,16,32,32,32,1,1,1,3> start = 17863,end = 170229,total = 152366
SplitFilterFloatPktStream<f,16,1224,216,72>::filter8 start = 17872,end = 170244,total = 152372
ConvHx4ReluPktStream<6,36,32,32,1,1,1,16,16,3,3,1,1> start = 17852,end = 248902,total = 231050
ConvHx4ReluPktStream<6,36,32,32,1,1,1,16,16,3,3,1,1> start = 17853,end = 250005,total = 232152
ConcatFloatStream<f,16,128,128,256> start = 17852,end = 250014,total = 232162
ConvHx4ReluPktStream<6,36,32,32,1,1,1,16,16,3,3,1,1> start = 17855,end = 251107,total = 233252
ConvHx4ReluPktStream<6,36,32,32,1,1,1,16,16,3,3,1,1> start = 17856,end = 252216,total = 234360
ConcatFloatStream<f,16,128,128,256> start = 17855,end = 252225,total = 234370
ConcatFloatStream<f,16,256,256,512> start = 17856,end = 252237,total = 234381
ConvHx4ReluPktStream<6,36,32,32,1,1,1,16,16,3,3,1,1> start = 17860,end = 253314,total = 235454
ConvHx4ReluPktStream<6,36,32,32,1,1,1,16,16,3,3,1,1> start = 17861,end = 254417,total = 236556
ConcatFloatStream<f,16,128,128,256> start = 17860,end = 254426,total = 236566
ConvHx4ReluPktStream<6,36,32,32,1,1,1,16,16,3,3,1,1> start = 17864,end = 255520,total = 237656
ConvHx4ReluPktStream<6,36,32,32,1,1,1,16,16,3,3,1,1> start = 17865,end = 256324,total = 238459
ConcatFloatStream<f,16,128,128,256> start = 17864,end = 256333,total = 238469
ConcatFloatStream<f,16,256,256,512> start = 17863,end = 256344,total = 238481
ConcatFloatStream<f,16,512,512,1024> start = 17862,end = 256354,total = 238492
Pad2DStreamFloat<f,16,32,32,32,1,1,1,3> start = 17875,end = 256409,total = 238534
SplitFilterFloatPktStream<f,16,1224,216,72>::filter8 start = 17886,end = 256422,total = 238536
ConvHx4ReluPktStream<6,36,32,32,1,1,1,16,16,3,3,1,0> start = 17868,end = 337024,total = 319156
ConvHx4ReluPktStream<6,36,32,32,1,1,1,16,16,3,3,1,0> start = 17869,end = 337265,total = 319396
ConcatFloatStream<f,16,128,128,256> start = 17868,end = 337274,total = 319406
ConvHx4ReluPktStream<6,36,32,32,1,1,1,16,16,3,3,1,0> start = 17871,end = 337511,total = 319640
ConvHx4ReluPktStream<6,36,32,32,1,1,1,16,16,3,3,1,0> start = 17872,end = 338022,total = 320150
ConcatFloatStream<f,16,128,128,256> start = 17871,end = 338031,total = 320160
ConcatFloatStream<f,16,256,256,512> start = 17871,end = 338041,total = 320170
ConvHx4ReluPktStream<6,36,32,32,1,1,1,16,16,3,3,1,0> start = 17874,end = 339113,total = 321239
ConvHx4ReluPktStream<6,36,32,32,1,1,1,16,16,3,3,1,0> start = 17875,end = 340214,total = 322339
ConcatFloatStream<f,16,128,128,256> start = 17874,end = 340223,total = 322349
ConvHx4ReluPktStream<6,36,32,32,1,1,1,16,16,3,3,1,0> start = 17869,end = 341324,total = 323455
ConvHx4ReluPktStream<6,36,32,32,1,1,1,16,16,3,3,1,0> start = 17870,end = 342078,total = 324208
ConcatFloatStream<f,16,128,128,256> start = 17869,end = 342087,total = 324218
ConcatFloatStream<f,16,256,256,512> start = 17872,end = 342096,total = 324224
ConcatFloatStream<f,16,512,512,1024> start = 17870,end = 342104,total = 324234
AddFloat<f,16384,1> start = 17863,end = 342133,total = 324270
SplitFilterFloatPktStream<f,16,1024,224,-32>::filter4 start = 17835,end = 342163,total = 324328
Pad2DStreamFloat<f,16,32,32,32,0,1,0,4> start = 17844,end = 342200,total = 324356
SplitFilterFloatPktStream<f,16,1188,180,36>::filter8 start = 17855,end = 342217,total = 324362
Conv1x1ReluStream<7,32,16,16,2,2,1,16,32,1,1,1,0> start = 17828,end = 367533,total = 349705
Conv1x1ReluStream<7,32,16,16,2,2,1,16,32,1,1,1,0> start = 17829,end = 368020,total = 350191
ConcatFloatStream<f,32,64,64,128> start = 17828,end = 368031,total = 350203
Conv1x1ReluStream<7,32,16,16,2,2,1,16,32,1,1,1,0> start = 17831,end = 369918,total = 352087
Conv1x1ReluStream<7,32,16,16,2,2,1,16,32,1,1,1,0> start = 17832,end = 371833,total = 354001
ConcatFloatStream<f,32,64,64,128> start = 17831,end = 371842,total = 354011
ConcatFloatStream<f,32,128,128,256> start = 17830,end = 371853,total = 354023
ConvHx4ReluPktStream<5,36,16,16,2,2,1,16,32,3,3,1,1> start = 17849,end = 428315,total = 410466
ConvHx4ReluPktStream<5,36,16,16,2,2,1,16,32,3,3,1,1> start = 17850,end = 428554,total = 410704
ConcatFloatStream<f,32,32,32,64> start = 17849,end = 428557,total = 410708
ConvHx4ReluPktStream<5,36,16,16,2,2,1,16,32,3,3,1,1> start = 17852,end = 428799,total = 410947
ConvHx4ReluPktStream<5,36,16,16,2,2,1,16,32,3,3,1,1> start = 17853,end = 429314,total = 411461
ConcatFloatStream<f,32,32,32,64> start = 17852,end = 429317,total = 411465
ConcatFloatStream<f,32,64,64,128> start = 17847,end = 429328,total = 411481
ConvHx4ReluPktStream<5,36,16,16,2,2,1,16,32,3,3,1,1> start = 17842,end = 430398,total = 412556
ConvHx4ReluPktStream<5,36,16,16,2,2,1,16,32,3,3,1,1> start = 17843,end = 431500,total = 413657
ConcatFloatStream<f,32,32,32,64> start = 17842,end = 431503,total = 413661
ConvHx4ReluPktStream<5,36,16,16,2,2,1,16,32,3,3,1,1> start = 17843,end = 432609,total = 414766
ConvHx4ReluPktStream<5,36,16,16,2,2,1,16,32,3,3,1,1> start = 17844,end = 433309,total = 415465
ConcatFloatStream<f,32,32,32,64> start = 17843,end = 433312,total = 415469
ConcatFloatStream<f,32,64,64,128> start = 17839,end = 433321,total = 415482
ConcatFloatStream<f,32,128,128,256> start = 17838,end = 433331,total = 415493
Pad2DStreamFloat<f,32,16,16,16,1,1,1,3> start = 17836,end = 433364,total = 415528
SplitFilterFloatPktStream<f,32,360,120,40>::filter4 start = 17842,end = 433376,total = 415534
ConvHx4ReluPktStream<6,20,16,16,1,1,1,32,32,3,3,1,0> start = 17835,end = 603719,total = 585884
ConvHx4ReluPktStream<6,20,16,16,1,1,1,32,32,3,3,1,0> start = 17836,end = 603864,total = 586028
ConcatFloatStream<f,32,64,64,128> start = 17835,end = 603873,total = 586038
ConvHx4ReluPktStream<6,20,16,16,1,1,1,32,32,3,3,1,0> start = 17838,end = 605308,total = 587470
ConvHx4ReluPktStream<6,20,16,16,1,1,1,32,32,3,3,1,0> start = 17839,end = 606371,total = 588532
ConcatFloatStream<f,32,64,64,128> start = 17838,end = 606380,total = 588542
ConcatFloatStream<f,32,128,128,256> start = 17838,end = 606390,total = 588552
AddFloat<f,8192,1> start = 17837,end = 606415,total = 588578
SplitFilterFloatPktStream<f,32,256,112,-16>::filter2 start = 17882,end = 606448,total = 588566
Pad2DStreamFloat<f,32,16,16,16,0,1,0,4> start = 17878,end = 606473,total = 588595
SplitFilterFloatPktStream<f,32,340,100,20>::filter4 start = 17893,end = 606490,total = 588597
Conv1x1ReluStream<7,16,8,8,2,2,1,32,64,1,1,1,0> start = 17881,end = 661748,total = 643867
Conv1x1ReluStream<7,16,8,8,2,2,1,32,64,1,1,1,0> start = 17882,end = 664253,total = 646371
ConcatFloatStream<f,64,32,32,64> start = 17881,end = 664264,total = 646383
ConvHx4ReluPktStream<5,20,8,8,2,2,1,32,64,3,3,1,1> start = 17884,end = 793172,total = 775288
ConvHx4ReluPktStream<5,20,8,8,2,2,1,32,64,3,3,1,1> start = 17885,end = 793316,total = 775431
ConcatFloatStream<f,64,16,16,32> start = 17884,end = 793317,total = 775433
ConvHx4ReluPktStream<5,20,8,8,2,2,1,32,64,3,3,1,1> start = 17887,end = 794763,total = 776876
ConcatFloatStream<f,64,16,16,32> start = 17887,end = 795786,total = 777899
ConvHx4ReluPktStream<5,20,8,8,2,2,1,32,64,3,3,1,1> start = 17888,end = 795787,total = 777899
ConcatFloatStream<f,64,32,32,64> start = 17888,end = 795801,total = 777913
Pad2DStreamFloat<f,64,8,8,8,1,1,1,3> start = 17889,end = 795848,total = 777959
SplitFilterFloatPktStream<f,64,120,48,24>::filter4 start = 17902,end = 795905,total = 778003
ConvHx4ReluPktStream<4,12,8,8,1,1,1,64,64,3,3,1,0> start = 17897,end = 1004153,total = 986256
ConcatFloatStream<f,64,16,16,32> start = 17897,end = 1005174,total = 987277
ConvHx4ReluPktStream<4,12,8,8,1,1,1,64,64,3,3,1,0> start = 17898,end = 1005175,total = 987277
ConvHx4ReluPktStream<4,12,8,8,1,1,1,64,64,3,3,1,0> start = 17900,end = 1005238,total = 987338
ConcatFloatStream<f,64,16,16,32> start = 17900,end = 1005263,total = 987363
ConvHx4ReluPktStream<4,12,8,8,1,1,1,64,64,3,3,1,0> start = 17901,end = 1005264,total = 987363
ConcatFloatStream<f,64,32,32,64> start = 17895,end = 1005277,total = 987382
AddFloat<f,4096,1> start = 17891,end = 1005304,total = 987413
AvgpoolScalarBCHW<8,8,1,1,1,64,8,8> start = 1005324,end = 1014953,total = 9629
GemmReluMKKN<1,64,12,0> start = 1015172,end = 1015439,total = 267
SoftmaxScalar<1,10,12> start = 1015602,end = 1046519,total = 30917

Checking 1/12: k001conv_goldenout_shape1x16x32x32.txt                                           against k001conv_goldenout_shape1x16x32x32.txt                  TEST: OK!
Checking 2/12: k003conv_goldenout_shape1x16x32x32.txt                                           against k003conv_goldenout_shape1x16x32x32.txt                  TEST: OK!
Checking 3/12: k006add_goldenout_shape1x16x32x32.txt                                            against k006add_goldenout_shape1x16x32x32.txt                   TEST: OK!
Checking 4/12: k008conv_goldenout_shape1x32x16x16.txt                                           against k008conv_goldenout_shape1x32x16x16.txt                  TEST: OK!
Checking 5/12: k009conv_goldenout_shape1x32x16x16.txt                                           against k009conv_goldenout_shape1x32x16x16.txt                  TEST: OK!
Checking 6/12: k012add_goldenout_shape1x32x16x16.txt                                            against k012add_goldenout_shape1x32x16x16.txt                   TEST: OK!
Checking 7/12: k014conv_goldenout_shape1x64x8x8.txt                                             against k014conv_goldenout_shape1x64x8x8.txt                    TEST: OK!
Checking 8/12: k015conv_goldenout_shape1x64x8x8.txt                                             against k015conv_goldenout_shape1x64x8x8.txt                    TEST: OK!
Checking 9/12: k018add_goldenout_shape1x64x8x8.txt                                              against k018add_goldenout_shape1x64x8x8.txt                     TEST: FAILED! Only 4093/4096 passed.
Max absolute difference: 0.00010681100000020649
Max relative difference: 0.0029229214921292017
Checking 10/12: k020pool_goldenout_shape1x64x1x1.txt                                             against k020pool_goldenout_shape1x64x1x1.txt                   TEST: OK!
Checking 11/12: k022gemm_goldenout_shape1x10.txt                                                 against k022gemm_goldenout_shape1x10.txt                       TEST: OK!
Checking 12/12: k024softmax_goldenout_shape1x10.txt                                              against k024softmax_goldenout_shape1x10.txt                    TEST: OK!
```

### tiny_ic_int8
Previous run with int8 only starts 6835 ends 438288.
```
TransposeScalarBHWC2BCHW<f,1,32,32,3> start = 6998,end = 13393,total = 6395
QuantizeLinearFmulStream<h,96,32,32> start = 6971,end = 19689,total = 12718
Pad2DStreamInt8<h,3,32,32,32,1,1,1,15> start = 6975,end = 19762,total = 12787
QLinearConvHx4Stream<h,h,34,48,32,32,1,1,1,3,16,3,3,1> start = 19777,end = 57779,total = 38002
Pad2DStreamInt8<h,16,32,32,32,1,1,1,15> start = 6964,end = 57852,total = 50888
SplitFilterInt8PktStream<h,16,1632,864,96>::filter2 start = 6965,end = 57881,total = 50916
QLinearConvHx4PktStream<h,h,18,48,32,32,1,1,1,16,16,3,3,1> start = 7023,end = 122976,total = 115953
ConcatInt8Stream<h,16,512,512,1024> start = 6964,end = 124010,total = 117046
QLinearConvHx4PktStream<h,h,18,48,32,32,1,1,1,16,16,3,3,1> start = 7024,end = 124006,total = 116982
Pad2DStreamInt8<h,16,32,32,32,1,1,1,15> start = 6956,end = 124084,total = 117128
SplitFilterInt8PktStream<h,16,1632,864,96>::filter2 start = 6953,end = 124115,total = 117162
QLinearConvHx4PktStream<h,h,18,48,32,32,1,1,1,16,16,3,3,1> start = 7007,end = 189232,total = 182225
ConcatInt8Stream<h,16,512,512,1024> start = 6948,end = 190243,total = 183295
QLinearConvHx4PktStream<h,h,18,48,32,32,1,1,1,16,16,3,3,1> start = 7008,end = 190239,total = 183231
QLinearAddInt8<h,16384,0> start = 7016,end = 190274,total = 183258
Pad2DStreamInt8<h,16,32,32,32,0,1,0,16> start = 6947,end = 190356,total = 183409
SplitFilterInt8PktStream<h,16,1584,816,48>::filter2 start = 6943,end = 190389,total = 183446
QLinearConv1x1Stream<h,h,32,32,16,16,2,2,1,16,32,1,1,1> start = 190300,end = 248756,total = 58456
QLinearConvHx4PktStream<h,h,17,48,16,16,2,2,1,16,32,3,3,1> start = 6991,end = 265536,total = 258545
ConcatInt8Stream<h,32,128,128,256> start = 6932,end = 266532,total = 259600
QLinearConvHx4PktStream<h,h,17,48,16,16,2,2,1,16,32,3,3,1> start = 6992,end = 266533,total = 259541
Pad2DStreamInt8<h,32,16,16,16,1,1,1,15> start = 6926,end = 266596,total = 259670
SplitFilterInt8PktStream<h,32,576,320,64>::filter2 start = 6930,end = 266621,total = 259691
QLinearConvHx4PktStream<h,h,10,32,16,16,1,1,1,32,32,3,3,1> start = 6988,end = 339780,total = 332792
ConcatInt8Stream<h,32,128,128,256> start = 6929,end = 340651,total = 333722
QLinearConvHx4PktStream<h,h,10,32,16,16,1,1,1,32,32,3,3,1> start = 6989,end = 340645,total = 333656
QLinearAddInt8<h,8192,0> start = 6981,end = 340683,total = 333702
Pad2DStreamInt8<h,32,16,16,16,0,1,0,16> start = 6909,end = 340767,total = 333858
SplitFilterInt8PktStream<h,32,544,288,32>::filter2 start = 6913,end = 340793,total = 333880
QLinearConv1x1Stream<h,h,16,16,8,16,2,2,1,32,64,1,1,1> start = 340704,end = 431962,total = 91258
QLinearConvHx4PktStream<h,h,9,32,8,16,2,2,1,32,64,3,3,1> start = 6968,end = 491939,total = 484971
ConcatInt8Stream<h,64,64,64,128> start = 6909,end = 492732,total = 485823
QLinearConvHx4PktStream<h,h,9,32,8,16,2,2,1,32,64,3,3,1> start = 6969,end = 492733,total = 485764
Pad2DStreamInt8<h,64,8,8,16,1,1,1,7> start = 6905,end = 492798,total = 485893
QLinearAddInt8<h,4096,0> start = 6978,end = 622965,total = 615987
QLinearConvHx4Stream<h,h,10,16,8,16,1,1,1,64,64,3,3,1> start = 492814,end = 753022,total = 260208
QLinearAddInt8<h,4096,0> start = 623081,end = 753045,total = 129964
QLinearAvgpoolScalarBCHW<8,16,1,1,1,64,8,8> start = 753059,end = 770552,total = 17493
QgemmStream<h,h,1,64,16> start = 6980,end = 770598,total = 763618
Pad2DStreamInt8<h,1,1,10,16,0,0,0,6> start = 6920,end = 770645,total = 763725
QLinearSoftmaxSingleaxis<h, 1,10,16> start = 770658,end = 770830,total = 172
DequantizeLinear<h,1,16,12> start = 770849,end = 770913,total = 64

Checking 1/11: k000transpose_goldenout_shape1x3x32x32.txt                                       against k000transpose_goldenout_shape1x3x32x32.txt      	TEST: OK!
Checking 2/11: k002qlinearconv_goldenout_shape1x16x32x32.txt                                    against k002qlinearconv_goldenout_shape1x16x32x32.txt   	TEST: FAILED! Only 16383/16384 passed.
Max absolute difference: 1.0
Max relative difference: 0.0625
Checking 3/11: k003qlinearconv_goldenout_shape1x16x32x32.txt                                    against k003qlinearconv_goldenout_shape1x16x32x32.txt   	TEST: FAILED! Only 16374/16384 passed.
Max absolute difference: 1.0
Max relative difference: 1.0
Checking 4/11: k005qlinearadd_goldenout_shape1x16x32x32.txt                                     against k005qlinearadd_goldenout_shape1x16x32x32.txt    	TEST: FAILED! Only 16336/16384 passed.
Max absolute difference: 2.0
Max relative difference: 1.0
Checking 5/11: k007qlinearconv_goldenout_shape1x32x16x16.txt                                    against k007qlinearconv_goldenout_shape1x32x16x16.txt   	TEST: FAILED! Only 8132/8192 passed.
Max absolute difference: 1.0
Max relative difference: 0.012195121951219513
Checking 6/11: k009qlinearadd_goldenout_shape1x32x16x16.txt                                     against k009qlinearadd_goldenout_shape1x32x16x16.txt    	TEST: FAILED! Only 7944/8192 passed.
Max absolute difference: 3.0
Max relative difference: 2.0
Checking 7/11: k013qlinearadd_goldenout_shape1x64x8x8.txt                                       against k013qlinearadd_goldenout_shape1x64x8x8.txt      	TEST: FAILED! Only 3690/4096 passed.
Max absolute difference: 4.0
Max relative difference: 3.0
Checking 8/11: k014qlinearpool_goldenout_shape1x64x1x1.txt                                      against k014qlinearpool_goldenout_shape1x64x1x1.txt     	TEST: FAILED! Only 39/64 passed.
Max absolute difference: 2.0
Max relative difference: 0.08333333333333333
Checking 9/11: k016qgemm_goldenout_shape1x10.txt                                                against k016qgemm_goldenout_shape1x10.txt               	TEST: FAILED! Only 6/10 passed.
Max absolute difference: 1.0
Max relative difference: 0.016129032258064516
Checking 10/11: k017qlinearsoftmax_goldenout_shape1x10.txt                                       against k017qlinearsoftmax_goldenout_shape1x10.txt      	TEST: FAILED! Only 8/10 passed.
Max absolute difference: 10.0
Max relative difference: 0.14285714285714285
Checking 11/11: k018dequantizeLinear_goldenout_shape1x10.txt                                     against k018dequantizeLinear_goldenout_shape1x10.txt    	TEST: FAILED! Only 8/10 passed.
Max absolute difference: 0.0390625
Max relative difference: 0.14285714285714285
```

### tiny_kws
Note need ~26gb ram and ~2h for aiesimulator. Previous run starts 8132 ends 823206.
```
Pad2DStreamFloat<f,1,49,10,16,4,5,1,5> start = 7736,end = 8691,total = 955
ConvHx4ReluStream<58,16,5,8,2,2,1,1,64,10,4,1,1> start = 8711,end = 312059,total = 303348
Pad2DStreamFloat<f,64,25,5,8,1,1,1,2> start = 7747,end = 312071,total = 304324
SplitFilterFloatPktStream<f,64,216,56,16>::filter5 start = 7761,end = 312087,total = 304326
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7759,end = 323070,total = 315311
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7760,end = 324004,total = 316244
ConcatFloatStream<f,64,40,40,80> start = 7759,end = 324016,total = 316257
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7762,end = 324943,total = 317181
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7763,end = 325877,total = 318114
ConcatFloatStream<f,64,40,40,80> start = 7762,end = 325889,total = 318127
ConcatFloatStream<f,64,80,80,160> start = 7757,end = 325900,total = 318143
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7753,end = 326264,total = 318511
ConcatFloatStream<f,64,160,40,200> start = 7753,end = 326503,total = 318750
Pad2DStreamFloat<f,64,25,5,8,0,0,0,3> start = 7748,end = 326513,total = 318765
SplitFilterFloatPktStream<f,64,200,40,0>::filter5 start = 7754,end = 330395,total = 322641
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7744,end = 406811,total = 399067
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7745,end = 406870,total = 399125
ConcatFloatStream<f,64,40,40,80> start = 7744,end = 406879,total = 399135
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7747,end = 406932,total = 399185
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7748,end = 406993,total = 399245
ConcatFloatStream<f,64,40,40,80> start = 7747,end = 407002,total = 399255
ConcatFloatStream<f,64,80,80,160> start = 7746,end = 407013,total = 399267
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7742,end = 407038,total = 399296
ConcatFloatStream<f,64,160,40,200> start = 7741,end = 407070,total = 399329
Pad2DStreamFloat<f,64,25,5,8,1,1,1,2> start = 7738,end = 407086,total = 399348
SplitFilterFloatPktStream<f,64,216,56,16>::filter5 start = 7738,end = 407119,total = 399381
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7728,end = 421083,total = 413355
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7729,end = 421139,total = 413410
ConcatFloatStream<f,64,40,40,80> start = 7728,end = 421151,total = 413423
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7731,end = 421198,total = 413467
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7732,end = 421256,total = 413524
ConcatFloatStream<f,64,40,40,80> start = 7731,end = 421268,total = 413537
ConcatFloatStream<f,64,80,80,160> start = 7726,end = 421279,total = 413553
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7727,end = 421298,total = 413571
ConcatFloatStream<f,64,160,40,200> start = 7723,end = 421562,total = 413839
Pad2DStreamFloat<f,64,25,5,8,0,0,0,3> start = 7723,end = 421571,total = 413848
SplitFilterFloatPktStream<f,64,200,40,0>::filter5 start = 7731,end = 425455,total = 417724
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7723,end = 501873,total = 494150
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7724,end = 501932,total = 494208
ConcatFloatStream<f,64,40,40,80> start = 7723,end = 501941,total = 494218
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7726,end = 501990,total = 494264
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7727,end = 502051,total = 494324
ConcatFloatStream<f,64,40,40,80> start = 7726,end = 502060,total = 494334
ConcatFloatStream<f,64,80,80,160> start = 7727,end = 502068,total = 494341
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7728,end = 502099,total = 494371
ConcatFloatStream<f,64,160,40,200> start = 7728,end = 502123,total = 494395
Pad2DStreamFloat<f,64,25,5,8,1,1,1,2> start = 7718,end = 502143,total = 494425
SplitFilterFloatPktStream<f,64,216,56,16>::filter5 start = 7701,end = 502196,total = 494495
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7691,end = 516160,total = 508469
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7692,end = 516216,total = 508524
ConcatFloatStream<f,64,40,40,80> start = 7691,end = 516228,total = 508537
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7694,end = 516275,total = 508581
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7695,end = 516333,total = 508638
ConcatFloatStream<f,64,40,40,80> start = 7694,end = 516345,total = 508651
ConcatFloatStream<f,64,80,80,160> start = 7693,end = 516356,total = 508663
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7689,end = 516376,total = 508687
ConcatFloatStream<f,64,160,40,200> start = 7690,end = 516639,total = 508949
Pad2DStreamFloat<f,64,25,5,8,0,0,0,3> start = 7693,end = 516649,total = 508956
SplitFilterFloatPktStream<f,64,200,40,0>::filter5 start = 7705,end = 520530,total = 512825
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7699,end = 596945,total = 589246
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7700,end = 597006,total = 589306
ConcatFloatStream<f,64,40,40,80> start = 7699,end = 597015,total = 589316
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7702,end = 597068,total = 589366
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7703,end = 597129,total = 589426
ConcatFloatStream<f,64,40,40,80> start = 7702,end = 597138,total = 589436
ConcatFloatStream<f,64,80,80,160> start = 7697,end = 597149,total = 589452
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7697,end = 597171,total = 589474
ConcatFloatStream<f,64,160,40,200> start = 7695,end = 597205,total = 589510
Pad2DStreamFloat<f,64,25,5,8,1,1,1,2> start = 7698,end = 597222,total = 589524
SplitFilterFloatPktStream<f,64,216,56,16>::filter5 start = 7711,end = 597250,total = 589539
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7709,end = 611216,total = 603507
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7710,end = 611272,total = 603562
ConcatFloatStream<f,64,40,40,80> start = 7709,end = 611284,total = 603575
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7712,end = 611331,total = 603619
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7713,end = 611389,total = 603676
ConcatFloatStream<f,64,40,40,80> start = 7712,end = 611401,total = 603689
ConcatFloatStream<f,64,80,80,160> start = 7707,end = 611412,total = 603705
ConvHx4ReluPktStream<7,8,5,8,1,1,1,64,64,3,3,64,1> start = 7708,end = 611431,total = 603723
ConcatFloatStream<f,64,160,40,200> start = 7704,end = 611695,total = 603991
Pad2DStreamFloat<f,64,25,5,8,0,0,0,3> start = 7704,end = 611704,total = 604000
SplitFilterFloatPktStream<f,64,200,40,0>::filter5 start = 7718,end = 615586,total = 607868
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7712,end = 692002,total = 684290
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7713,end = 692061,total = 684348
ConcatFloatStream<f,64,40,40,80> start = 7712,end = 692070,total = 684358
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7715,end = 692123,total = 684408
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7716,end = 692184,total = 684468
ConcatFloatStream<f,64,40,40,80> start = 7715,end = 692193,total = 684478
ConcatFloatStream<f,64,80,80,160> start = 7710,end = 692204,total = 684494
Conv1x1ReluStream<5,8,5,8,1,1,1,64,64,1,1,1,1> start = 7711,end = 692229,total = 684518
ConcatFloatStream<f,64,160,40,200> start = 7707,end = 692261,total = 684554
SplitScalar<f,1,12800,3200,0>::filter4 start = 7767,end = 805022,total = 797255
AvgpoolScalarBCHW<25,8,1,1,1,16,25,5> start = 805156,end = 809537,total = 4381
AvgpoolScalarBCHW<25,8,1,1,1,16,25,5> start = 805160,end = 809541,total = 4381
AvgpoolScalarBCHW<25,8,1,1,1,16,25,5> start = 805164,end = 809545,total = 4381
AvgpoolScalarBCHW<25,8,1,1,1,16,25,5> start = 805168,end = 809549,total = 4381
ConcatFloatStream<f,1,16,16,32> start = 7700,end = 809703,total = 802003
ConcatFloatStream<f,1,16,16,32> start = 7702,end = 809716,total = 802014
ConcatFloatStream<f,1,32,32,64> start = 7701,end = 809752,total = 802051
GemmReluMKKN<1,64,12,0> start = 809767,end = 810034,total = 267
SoftmaxScalar<1,12,12> start = 810192,end = 847335,total = 37143
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
Note need ~26gb ram and ~1h for aiesimulator. Previous run on int8 only starts 5725 ends 330087.
```
QuantizeLinearFmulStream<h,49,12,16> start = 5305,end = 6870,total = 1565
Pad2DStreamInt8<h,1,49,10,16,4,5,1,5> start = 5304,end = 6948,total = 1644
QLinearConvHx4Stream<h,h,58,16,5,16,2,2,1,1,64,10,4,1> start = 6969,end = 111150,total = 104181
Pad2DStreamInt8<h,64,25,5,16,1,1,1,10> start = 5308,end = 111216,total = 105908
SplitFilterInt8PktStream<h,64,432,112,32>::filter5 start = 5320,end = 111234,total = 105914
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5373,end = 116885,total = 111512
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5374,end = 117196,total = 111822
ConcatInt8Stream<h,64,80,80,160> start = 5314,end = 117206,total = 111892
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5376,end = 117508,total = 112132
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5377,end = 117876,total = 112499
ConcatInt8Stream<h,64,80,80,160> start = 5317,end = 117886,total = 112569
ConcatInt8Stream<h,64,160,160,320> start = 5312,end = 117899,total = 112587
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5371,end = 117909,total = 112538
ConcatInt8Stream<h,64,320,80,400> start = 5308,end = 117937,total = 112629
Pad2DStreamInt8<h,64,25,5,16,0,0,0,11> start = 5314,end = 147850,total = 142536
SplitFilterInt8PktStream<h,64,400,80,0>::filter5 start = 5328,end = 147873,total = 142545
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5387,end = 190903,total = 185516
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5388,end = 191012,total = 185624
ConcatInt8Stream<h,64,80,80,160> start = 5328,end = 191022,total = 185694
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5390,end = 191128,total = 185738
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5391,end = 191218,total = 185827
ConcatInt8Stream<h,64,80,80,160> start = 5331,end = 191228,total = 185897
ConcatInt8Stream<h,64,160,160,320> start = 5326,end = 191241,total = 185915
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5379,end = 191299,total = 185920
ConcatInt8Stream<h,64,320,80,400> start = 5324,end = 191311,total = 185987
Pad2DStreamInt8<h,64,25,5,16,1,1,1,10> start = 5330,end = 196622,total = 191292
SplitFilterInt8PktStream<h,64,432,112,32>::filter5 start = 5338,end = 196640,total = 191302
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5391,end = 202786,total = 197395
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5392,end = 202932,total = 197540
ConcatInt8Stream<h,64,80,80,160> start = 5332,end = 202942,total = 197610
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5394,end = 203079,total = 197685
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5395,end = 203282,total = 197887
ConcatInt8Stream<h,64,80,80,160> start = 5335,end = 203292,total = 197957
ConcatInt8Stream<h,64,160,160,320> start = 5334,end = 203305,total = 197971
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5389,end = 203315,total = 197926
ConcatInt8Stream<h,64,320,80,400> start = 5330,end = 203343,total = 198013
Pad2DStreamInt8<h,64,25,5,16,0,0,0,11> start = 5335,end = 233255,total = 227920
SplitFilterInt8PktStream<h,64,400,80,0>::filter5 start = 5347,end = 233278,total = 227931
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5403,end = 276308,total = 270905
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5404,end = 276417,total = 271013
ConcatInt8Stream<h,64,80,80,160> start = 5344,end = 276427,total = 271083
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5406,end = 276529,total = 271123
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5407,end = 276621,total = 271214
ConcatInt8Stream<h,64,80,80,160> start = 5347,end = 276631,total = 271284
ConcatInt8Stream<h,64,160,160,320> start = 5342,end = 276644,total = 271302
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5398,end = 276704,total = 271306
ConcatInt8Stream<h,64,320,80,400> start = 5341,end = 276715,total = 271374
Pad2DStreamInt8<h,64,25,5,16,1,1,1,10> start = 5348,end = 282026,total = 276678
SplitFilterInt8PktStream<h,64,432,112,32>::filter5 start = 5359,end = 282044,total = 276685
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5406,end = 288192,total = 282786
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5407,end = 288336,total = 282929
ConcatInt8Stream<h,64,80,80,160> start = 5347,end = 288346,total = 282999
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5409,end = 288483,total = 283074
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5410,end = 288686,total = 283276
ConcatInt8Stream<h,64,80,80,160> start = 5350,end = 288696,total = 283346
ConcatInt8Stream<h,64,160,160,320> start = 5349,end = 288709,total = 283360
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5415,end = 288720,total = 283305
ConcatInt8Stream<h,64,320,80,400> start = 5355,end = 288746,total = 283391
Pad2DStreamInt8<h,64,25,5,16,0,0,0,11> start = 5355,end = 318656,total = 313301
SplitFilterInt8PktStream<h,64,400,80,0>::filter5 start = 5362,end = 318680,total = 313318
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5419,end = 361709,total = 356290
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5420,end = 361818,total = 356398
ConcatInt8Stream<h,64,80,80,160> start = 5360,end = 361828,total = 356468
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5422,end = 361934,total = 356512
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5423,end = 362024,total = 356601
ConcatInt8Stream<h,64,80,80,160> start = 5363,end = 362034,total = 356671
ConcatInt8Stream<h,64,160,160,320> start = 5358,end = 362047,total = 356689
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5413,end = 362106,total = 356693
ConcatInt8Stream<h,64,320,80,400> start = 5356,end = 362117,total = 356761
Pad2DStreamInt8<h,64,25,5,16,1,1,1,10> start = 5359,end = 367426,total = 362067
SplitFilterInt8PktStream<h,64,432,112,32>::filter5 start = 5373,end = 367446,total = 362073
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5422,end = 373593,total = 368171
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5423,end = 373737,total = 368314
ConcatInt8Stream<h,64,80,80,160> start = 5363,end = 373747,total = 368384
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5425,end = 373884,total = 368459
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5426,end = 374087,total = 368661
ConcatInt8Stream<h,64,80,80,160> start = 5366,end = 374097,total = 368731
ConcatInt8Stream<h,64,160,160,320> start = 5370,end = 374111,total = 368741
QLinearConvHx4PktStream<h,h,7,16,5,16,1,1,1,64,64,3,3,64> start = 5429,end = 374122,total = 368693
ConcatInt8Stream<h,64,320,80,400> start = 5369,end = 374146,total = 368777
Pad2DStreamInt8<h,64,25,5,16,0,0,0,11> start = 5372,end = 404057,total = 398685
SplitFilterInt8PktStream<h,64,400,80,0>::filter5 start = 5384,end = 404079,total = 398695
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5435,end = 447108,total = 441673
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5436,end = 447217,total = 441781
ConcatInt8Stream<h,64,80,80,160> start = 5376,end = 447227,total = 441851
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5438,end = 447327,total = 441889
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5439,end = 447419,total = 441980
ConcatInt8Stream<h,64,80,80,160> start = 5379,end = 447429,total = 442050
ConcatInt8Stream<h,64,160,160,320> start = 5376,end = 447440,total = 442064
QLinearConv1x1PktStream<h,h,5,16,5,16,1,1,1,64,64,1,1,1> start = 5432,end = 447506,total = 442074
ConcatInt8Stream<h,64,320,80,400> start = 5372,end = 447516,total = 442144
SplitInt8<h,1,25600,12800,0>::filter2 start = 5401,end = 447542,total = 442141
ConcatInt8Stream<h,1,32,32,64> start = 5366,end = 460351,total = 454985
QgemmStream<h,h,1,64,16> start = 460364,end = 460492,total = 128
Pad2DStreamInt8<h,1,1,12,16,0,0,0,4> start = 5350,end = 460546,total = 455196
7013 9885 1151330 54616 7013 156 353372 541979123 212647 4973 9 4411312 16783873 16783873 16783873 16783873 
QLinearSoftmaxSingleaxis<h, 1,12,16> start = 460558,end = 461685,total = 1127
DequantizeLinear<h,1,16,12> start = 461704,end = 461768,total = 64
QLinearAvgpoolScalarBCHW<25,16,1,1,1,64,25,5> start = 447672,end = 472973,total = 25301
QLinearAvgpoolScalarBCHW<25,16,1,1,1,64,25,5> start = 447676,end = 472977,total = 25301
core(s) are done executing
[INFO]: Fifo Guidance report generated ./AIESim_FIFO_Guidance.json
Exiting!
generate profile data for all cores
generate profile data for all cores
Stopping Simulator.

Info: /OSCI/SystemC: Simulation stopped by user.
IP-INFO: deleting ip PSIP_ps_i139 
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
Checking directories out: /home/home/onnx2versal/reports_dir/tiny_kws_int8/hw_emu/aiesimulator_output and data: /home/home/onnx2versal/data at tolerance @ rtol=1e-03, atol=1e-05
Checking 1/14: k001quantizelinear_goldenout_shape1x1x49x10.txt                                  against k001quantizelinear_goldenout_shape1x1x49x10.txt  TEST: OK!
Checking 2/14: k002qlinearconv_goldenout_shape1x64x25x5.txt                                     against k002qlinearconv_goldenout_shape1x64x25x5.txt     TEST: OK!
Checking 3/14: k003qlinearconv_goldenout_shape1x64x25x5.txt                                     against k003qlinearconv_goldenout_shape1x64x25x5.txt     TEST: OK!
Checking 4/14: k004qlinearconv_goldenout_shape1x64x25x5.txt                                     against k004qlinearconv_goldenout_shape1x64x25x5.txt     TEST: OK!
Checking 5/14: k005qlinearconv_goldenout_shape1x64x25x5.txt                                     against k005qlinearconv_goldenout_shape1x64x25x5.txt     TEST: FAILED! Only 7998/8000 passed.
Max absolute difference: 1.0
Max relative difference: 0.07692307692307693
Checking 6/14: k006qlinearconv_goldenout_shape1x64x25x5.txt                                     against k006qlinearconv_goldenout_shape1x64x25x5.txt     TEST: FAILED! Only 7990/8000 passed.
Max absolute difference: 1.0
Max relative difference: 0.3333333333333333
Checking 7/14: k007qlinearconv_goldenout_shape1x64x25x5.txt                                     against k007qlinearconv_goldenout_shape1x64x25x5.txt     TEST: FAILED! Only 7987/8000 passed.
Max absolute difference: 2.0
Max relative difference: 1.0
Checking 8/14: k008qlinearconv_goldenout_shape1x64x25x5.txt                                     against k008qlinearconv_goldenout_shape1x64x25x5.txt     TEST: FAILED! Only 7923/8000 passed.
Max absolute difference: 1.0
Max relative difference: 1.0
Checking 9/14: k009qlinearconv_goldenout_shape1x64x25x5.txt                                     against k009qlinearconv_goldenout_shape1x64x25x5.txt     TEST: FAILED! Only 7932/8000 passed.
Max absolute difference: 1.0
Max relative difference: 1.0
Checking 10/14: k010qlinearconv_goldenout_shape1x64x25x5.txt                                     against k010qlinearconv_goldenout_shape1x64x25x5.txt            TEST: FAILED! Only 7685/8000 passed.
Max absolute difference: 2.0
Max relative difference: 1.0
Checking 11/14: k011qlinearpool_goldenout_shape1x64x1x1.txt                                      against k011qlinearpool_goldenout_shape1x64x1x1.txt             TEST: FAILED! Only 56/64 passed.
Max absolute difference: 1.0
Max relative difference: 0.045454545454545456
Checking 12/14: k013qgemm_goldenout_shape1x12.txt                                                against k013qgemm_goldenout_shape1x12.txt                       TEST: FAILED! Only 11/12 passed.
Max absolute difference: 1.0
Max relative difference: 0.008928571428571428
Checking 13/14: k014qlinearsoftmax_goldenout_shape1x12.txt                                       against k014qlinearsoftmax_goldenout_shape1x12.txt              TEST: FAILED! Only 11/12 passed.
Max absolute difference: 1.0
Max relative difference: 0.003952569169960474
Checking 14/14: k015dequantizeLinear_goldenout_shape1x12.txt                                     against k015dequantizeLinear_goldenout_shape1x12.txt            TEST: FAILED! Only 11/12 passed.
Max absolute difference: 0.00390625
Max relative difference: 0.003952569169960474
```

### tiny_vww
```
```

### tiny_vww_int8
```
```