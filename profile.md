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
GemmReluMKKN<1,16,64,1> start = 1501,end = 1913,total = 412
GemmReluMKKN<1,64,32,1> start = 2123,end = 2641,total = 518
GemmReluMKKN<1,32,32,1> start = 2822,end = 3128,total = 306
GemmReluMKKN<1,32,8,0> start = 3303,end = 3419,total = 116
SoftmaxSingleaxis<1,5,8> start = 3568,end = 3708,total = 140

Checking 1/5: k000gemm_goldenout_shape1x64.txt                                                 against k000gemm_goldenout_shape1x64.txt                 TEST: OK!
Checking 2/5: k003gemm_goldenout_shape1x32.txt                                                 against k003gemm_goldenout_shape1x32.txt                 TEST: OK!
Checking 3/5: k006gemm_goldenout_shape1x32.txt                                                 against k006gemm_goldenout_shape1x32.txt                 TEST: OK!
Checking 4/5: k009gemm_goldenout_shape1x5.txt                                                  against k009gemm_goldenout_shape1x5.txt                  TEST: OK!
Checking 5/5: k011softmax_goldenout_shape1x5.txt                                               against k011softmax_goldenout_shape1x5.txt               TEST: FAILED! Only 1/5 passed.
Max absolute difference: 0.0006341933999999272
Max relative difference: 0.024902617338628467
```

### hls4ml_jettag_int8

```
QuantizeLinearFmulStream<a,1,16,16> start = 884,end = 953,total = 69
QgemmStream<a,a,1,16,64> start = 944,end = 1118,total = 174
QgemmStream<a,a,1,64,32> start = 946,end = 1242,total = 296
QgemmStream<a,a,1,32,32> start = 948,end = 1361,total = 413
QgemmStream<a,a,1,32,16> start = 950,end = 1432,total = 482
Pad2DStreamInt8<a,1,1,5,16,0,0,0,11> start = 893,end = 1480,total = 587
QLinearSoftmaxSingleaxis<a, 1,5,16> start = 1494,end = 1666,total = 172
DequantizeLinear<a,1,16,8> start = 1684,end = 1740,total = 56

Checking 1/7: k000quantizelinear_goldenout_shape1x16.txt                                       against k000quantizelinear_goldenout_shape1x16.txt       TEST: OK!
Checking 2/7: k001qgemm_goldenout_shape1x64.txt                                                against k001qgemm_goldenout_shape1x64.txt                TEST: OK!
Checking 3/7: k002qgemm_goldenout_shape1x32.txt                                                against k002qgemm_goldenout_shape1x32.txt                TEST: OK!
Checking 4/7: k003qgemm_goldenout_shape1x32.txt                                                against k003qgemm_goldenout_shape1x32.txt                TEST: OK!
Checking 5/7: k004qgemm_goldenout_shape1x5.txt                                                 against k004qgemm_goldenout_shape1x5.txt                 TEST: OK!
Checking 6/7: k005qlinearsoftmax_goldenout_shape1x5.txt                                        against k005qlinearsoftmax_goldenout_shape1x5.txt        TEST: OK!
Checking 7/7: k006dequantizeLinear_goldenout_shape1x5.txt                                      against k006dequantizeLinear_goldenout_shape1x5.txt      TEST: OK!
```

### lenet_mnist
Previous run starts 3221 ends 59766.
```
Pad2DStreamFloat<f,1,28,28,32,0,0,0,4> start = 2836,end = 3748,total = 912
ConvHx8ReluStream<28,32,24,24,1,1,1,1,6,5,5,1,1> start = 3768,end = 27956,total = 24188
Maxpool2x2FloatBCHW<24,24,12,16,1,6,2,2> start = 27963,end = 28870,total = 907
Pad2DStreamFloat<f,6,12,12,16,0,0,0,4> start = 2830,end = 30157,total = 27327
ConvHx8ReluStream<12,16,8,8,1,1,1,6,16,5,5,1,1> start = 30176,end = 61266,total = 31090
Maxpool2x2FloatBCHW<8,8,4,4,1,16,2,2> start = 61266,end = 61564,total = 298
GemmReluMKKN<1,256,16,1> start = 61963,end = 62869,total = 906
GemmReluMKKN<1,256,16,1> start = 61965,end = 62871,total = 906
GemmReluMKKN<1,256,16,1> start = 61965,end = 62871,total = 906
GemmReluMKKN<1,256,16,1> start = 61967,end = 62873,total = 906
GemmReluMKKN<1,256,16,1> start = 61968,end = 62874,total = 906
GemmReluMKKN<1,256,16,1> start = 61969,end = 62875,total = 906
GemmReluMKKN<1,256,16,1> start = 61970,end = 62876,total = 906
GemmReluMKKN<1,256,16,1> start = 61975,end = 62881,total = 906
ConcatFloatStream<f,1,16,16,32> start = 2834,end = 63034,total = 60200
ConcatFloatStream<f,1,16,16,32> start = 2837,end = 63037,total = 60200
ConcatFloatStream<f,1,16,16,32> start = 2828,end = 63039,total = 60211
ConcatFloatStream<f,1,16,16,32> start = 2826,end = 63046,total = 60220
ConcatFloatStream<f,1,32,32,64> start = 2833,end = 63087,total = 60254
ConcatFloatStream<f,1,32,32,64> start = 2827,end = 63088,total = 60261
ConcatFloatStream<f,1,64,64,120> start = 2831,end = 63168,total = 60337
GemmReluMKKN<1,120,32,1> start = 63177,end = 64073,total = 896
GemmReluMKKN<1,120,32,1> start = 63179,end = 64075,total = 896
GemmReluMKKN<1,120,32,1> start = 63179,end = 64075,total = 896
ConcatFloatStream<f,1,32,32,64> start = 2828,end = 64282,total = 61454
ConcatFloatStream<f,1,64,32,84> start = 2827,end = 64329,total = 61502
GemmReluScalarMKKN<1,84,10,1> start = 64334,end = 66236,total = 1902

Checking 1/7: k000conv_goldenout_shape1x6x24x24.txt                                            against k000conv_goldenout_shape1x6x24x24.txt            TEST: OK!
Checking 2/7: k002pool_goldenout_shape1x6x12x12.txt                                            against k002pool_goldenout_shape1x6x12x12.txt            TEST: OK!
Checking 3/7: k003conv_goldenout_shape1x16x8x8.txt                                             against k003conv_goldenout_shape1x16x8x8.txt             TEST: OK!
Checking 4/7: k005pool_goldenout_shape1x16x4x4.txt                                             against k005pool_goldenout_shape1x16x4x4.txt             TEST: OK!
Checking 5/7: k014gemm_goldenout_shape1x120.txt                                                against k014gemm_goldenout_shape1x120.txt                TEST: OK!
Checking 6/7: k016gemm_goldenout_shape1x84.txt                                                 against k016gemm_goldenout_shape1x84.txt                 TEST: OK!
Checking 7/7: k018gemm_goldenout_shape1x10.txt                                                 against k018gemm_goldenout_shape1x10.txt                 TEST: OK!
```

### lenet_mnist_int8
Previous run starts 3035 ends 24222.
```
QuantizeLinearFmulStream<a,28,28,32> start = 1574,end = 3324,total = 1750
Pad2DStreamInt8<a,1,28,28,32,0,0,0,4> start = 1573,end = 3365,total = 1792
QLinearConvHx6x8bitStream<a,a,28,32,24,32,1,1,1,1,6,5,5,1> start = 3385,end = 6303,total = 2918
Maxpool2x2Int8BCHW<24,32,12,16,1,6,2,2> start = 6323,end = 6652,total = 329
Pad2DStreamInt8<a,6,12,12,16,0,0,0,4> start = 1571,end = 8517,total = 6946
QLinearConvHx6x8bitStream<a,a,12,16,8,16,1,1,1,6,16,5,5,1> start = 8538,end = 21537,total = 12999
MaxpoolScalarBCHW<8,16,4,4,1,16,2,2> start = 21554,end = 25152,total = 3598
QgemmStream<a,a,1,256,64> start = 1689,end = 26386,total = 24697
QgemmStream<a,a,1,256,64> start = 1690,end = 26386,total = 24696
ConcatInt8Stream<a,1,64,64,128> start = 1569,end = 26413,total = 24844
QgemmStream<a,a,1,128,96> start = 1687,end = 27212,total = 25525
QgemmStream<a,a,1,96,16> start = 1685,end = 27266,total = 25581
DequantizeLinear<a,1,16,12> start = 27286,end = 27350,total = 64

Checking 1/9: k000quantizelinear_goldenout_shape1x1x28x28.txt                                  against k000quantizelinear_goldenout_shape1x1x28x28.txt  TEST: OK!
Checking 2/9: k001qlinearconv_goldenout_shape1x6x24x24.txt                                     against k001qlinearconv_goldenout_shape1x6x24x24.txt     TEST: OK!
Checking 3/9: k002pool_goldenout_shape1x6x12x12.txt                                            against k002pool_goldenout_shape1x6x12x12.txt            TEST: OK!
Checking 4/9: k003qlinearconv_goldenout_shape1x16x8x8.txt                                      against k003qlinearconv_goldenout_shape1x16x8x8.txt      TEST: OK!
Checking 5/9: k004pool_goldenout_shape1x16x4x4.txt                                             against k004pool_goldenout_shape1x16x4x4.txt             TEST: OK!
Checking 6/9: k006qgemm_goldenout_shape1x120.txt                                               against k006qgemm_goldenout_shape1x120.txt               TEST: OK!
Checking 7/9: k007qgemm_goldenout_shape1x84.txt                                                against k007qgemm_goldenout_shape1x84.txt                TEST: OK!
Checking 8/9: k008qgemm_goldenout_shape1x10.txt                                                against k008qgemm_goldenout_shape1x10.txt                TEST: OK!
Checking 9/9: k009dequantizeLinear_goldenout_shape1x10.txt                                     against k009dequantizeLinear_goldenout_shape1x10.txt     TEST: FAILED! Only 7/10 passed.
```

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
QuantizeLinearFmulStream<h,49,12,16> start = 5623,end = 7188,total = 1565
Pad2DStreamInt8<h,1,49,10,16,4,5,1,5> start = 5624,end = 7265,total = 1641
SplitFilterInt8PktStream<h,1,928,288,128>::filter5 start = 5631,end = 7353,total = 1722
QLinearConvHx4PktStream<h,a,18,16,5,16,2,2,1,1,64,10,4,1> start = 5678,end = 35515,total = 29837
ConcatInt8Stream<h,64,80,80,160> start = 5621,end = 35831,total = 30210
QLinearConvHx4PktStream<h,a,18,16,5,16,2,2,1,1,64,10,4,1> start = 5679,end = 35825,total = 30146
QLinearConvHx4PktStream<h,a,18,16,5,16,2,2,1,1,64,10,4,1> start = 5681,end = 36140,total = 30459
ConcatInt8Stream<h,64,80,80,160> start = 5624,end = 36262,total = 30638
QLinearConvHx4PktStream<h,a,18,16,5,16,2,2,1,1,64,10,4,1> start = 5682,end = 36256,total = 30574
ConcatInt8Stream<h,64,160,160,320> start = 5623,end = 36275,total = 30652
QLinearConvHx4PktStream<h,a,18,16,5,16,2,2,1,1,64,10,4,1> start = 5685,end = 36298,total = 30613
ConcatInt8Stream<h,64,320,80,400> start = 5627,end = 36306,total = 30679
Pad2DStreamInt8<h,64,25,5,16,1,1,1,10> start = 5628,end = 55797,total = 50169
SplitFilterInt8PktStream<h,64,432,112,32>::filter5 start = 5640,end = 55815,total = 50175
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5693,end = 58870,total = 53177
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5694,end = 59016,total = 53322
ConcatInt8Stream<h,64,80,80,160> start = 5634,end = 59026,total = 53392
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5696,end = 59157,total = 53461
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5697,end = 59360,total = 53663
ConcatInt8Stream<h,64,80,80,160> start = 5637,end = 59370,total = 53733
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5689,end = 59397,total = 53708
ConcatInt8Stream<h,64,160,160,320> start = 5632,end = 60945,total = 55313
ConcatInt8Stream<h,64,320,80,400> start = 5629,end = 62236,total = 56607
Pad2DStreamInt8<h,64,25,5,16,0,0,0,11> start = 5629,end = 92424,total = 86795
SplitFilterInt8PktStream<h,64,400,80,0>::filter5 start = 5659,end = 92455,total = 86796
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5711,end = 126716,total = 121005
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5712,end = 126825,total = 121113
ConcatInt8Stream<h,64,80,80,160> start = 5652,end = 126835,total = 121183
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5714,end = 126937,total = 121223
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5715,end = 127029,total = 121314
ConcatInt8Stream<h,64,80,80,160> start = 5655,end = 127039,total = 121384
ConcatInt8Stream<h,64,160,160,320> start = 5651,end = 127051,total = 121400
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5710,end = 127114,total = 121404
ConcatInt8Stream<h,64,320,80,400> start = 5653,end = 127125,total = 121472
Pad2DStreamInt8<h,64,25,5,16,1,1,1,10> start = 5656,end = 141067,total = 135411
SplitFilterInt8PktStream<h,64,432,112,32>::filter5 start = 5672,end = 141087,total = 135415
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5725,end = 144139,total = 138414
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5726,end = 144283,total = 138557
ConcatInt8Stream<h,64,80,80,160> start = 5666,end = 144293,total = 138627
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5728,end = 144430,total = 138702
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5729,end = 144633,total = 138904
ConcatInt8Stream<h,64,80,80,160> start = 5669,end = 144643,total = 138974
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5725,end = 144667,total = 138942
ConcatInt8Stream<h,64,160,160,320> start = 5665,end = 146217,total = 140552
ConcatInt8Stream<h,64,320,80,400> start = 5662,end = 147508,total = 141846
Pad2DStreamInt8<h,64,25,5,16,0,0,0,11> start = 5647,end = 177702,total = 172055
SplitFilterInt8PktStream<h,64,400,80,0>::filter5 start = 5647,end = 177728,total = 172081
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5696,end = 211988,total = 206292
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5697,end = 212097,total = 206400
ConcatInt8Stream<h,64,80,80,160> start = 5637,end = 212107,total = 206470
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5699,end = 212211,total = 206512
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5700,end = 212303,total = 206603
ConcatInt8Stream<h,64,80,80,160> start = 5640,end = 212313,total = 206673
ConcatInt8Stream<h,64,160,160,320> start = 5639,end = 212326,total = 206687
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5702,end = 212389,total = 206687
ConcatInt8Stream<h,64,320,80,400> start = 5642,end = 212399,total = 206757
Pad2DStreamInt8<h,64,25,5,16,1,1,1,10> start = 5642,end = 226336,total = 220694
SplitFilterInt8PktStream<h,64,432,112,32>::filter5 start = 5626,end = 226366,total = 220740
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5677,end = 229420,total = 223743
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5678,end = 229564,total = 223886
ConcatInt8Stream<h,64,80,80,160> start = 5618,end = 229574,total = 223956
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5680,end = 229707,total = 224027
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5681,end = 229910,total = 224229
ConcatInt8Stream<h,64,80,80,160> start = 5621,end = 229920,total = 224299
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5676,end = 229947,total = 224271
ConcatInt8Stream<h,64,160,160,320> start = 5616,end = 231495,total = 225879
ConcatInt8Stream<h,64,320,80,400> start = 5613,end = 232786,total = 227173
Pad2DStreamInt8<h,64,25,5,16,0,0,0,11> start = 5606,end = 262976,total = 257370
SplitFilterInt8PktStream<h,64,400,80,0>::filter5 start = 5612,end = 262999,total = 257387
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5665,end = 297258,total = 291593
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5666,end = 297369,total = 291703
ConcatInt8Stream<h,64,80,80,160> start = 5606,end = 297379,total = 291773
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5668,end = 297483,total = 291815
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5669,end = 297575,total = 291906
ConcatInt8Stream<h,64,80,80,160> start = 5609,end = 297585,total = 291976
ConcatInt8Stream<h,64,160,160,320> start = 5608,end = 297598,total = 291990
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5663,end = 297658,total = 291995
ConcatInt8Stream<h,64,320,80,400> start = 5604,end = 297668,total = 292064
Pad2DStreamInt8<h,64,25,5,16,1,1,1,10> start = 5597,end = 311609,total = 306012
SplitFilterInt8PktStream<h,64,432,112,32>::filter5 start = 5608,end = 311627,total = 306019
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5661,end = 314679,total = 309018
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5662,end = 314823,total = 309161
ConcatInt8Stream<h,64,80,80,160> start = 5602,end = 314833,total = 309231
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5664,end = 314970,total = 309306
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5665,end = 315173,total = 309508
ConcatInt8Stream<h,64,80,80,160> start = 5605,end = 315183,total = 309578
QLinearConvHx4PktStream<h,a,7,16,5,16,1,1,1,64,64,3,3,64> start = 5657,end = 315208,total = 309551
ConcatInt8Stream<h,64,160,160,320> start = 5600,end = 316758,total = 311158
ConcatInt8Stream<h,64,320,80,400> start = 5598,end = 318048,total = 312450
Pad2DStreamInt8<h,64,25,5,16,0,0,0,11> start = 5596,end = 348236,total = 342640
SplitFilterInt8PktStream<h,64,400,80,0>::filter5 start = 5604,end = 348258,total = 342654
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5657,end = 382517,total = 376860
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5658,end = 382628,total = 376970
ConcatInt8Stream<h,64,80,80,160> start = 5598,end = 382638,total = 377040
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5660,end = 382742,total = 377082
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5661,end = 382834,total = 377173
ConcatInt8Stream<h,64,80,80,160> start = 5601,end = 382844,total = 377243
ConcatInt8Stream<h,64,160,160,320> start = 5600,end = 382857,total = 377257
QLinearConv1x1PktStream<h,a,5,16,5,16,1,1,1,64,64,1,1,1> start = 5655,end = 382917,total = 377262
ConcatInt8Stream<h,64,320,80,400> start = 5599,end = 382930,total = 377331
SplitInt8<h,1,25600,12800,0>::filter2 start = 5642,end = 382962,total = 377320
ConcatInt8Stream<h,1,32,32,64> start = 5611,end = 395769,total = 390158
QgemmStream<h,a,1,64,16> start = 5669,end = 395850,total = 390181
Pad2DStreamInt8<h,1,1,12,16,0,0,0,4> start = 5609,end = 395897,total = 390288
QLinearSoftmaxSingleaxis<h, 1,12,16> start = 395911,end = 396084,total = 173
DequantizeLinear<h,1,16,12> start = 396104,end = 396168,total = 64
QLinearAvgpoolScalarBCHW<25,16,1,1,1,64,25,5> start = 383092,end = 408392,total = 25300
QLinearAvgpoolScalarBCHW<25,16,1,1,1,64,25,5> start = 383096,end = 408395,total = 25299
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
TransposeScalarBHWC2BCHW<f,1,24,96,3> start = 20243,end = 28213,total = 7970
TransposeScalarBHWC2BCHW<f,1,24,96,3> start = 27163,end = 35133,total = 7970
SplitFilterFloatStreamTwice<f,1,27648,6912,0>::filter2 start = 13282,end = 40987,total = 27705
SplitFilterFloatStreamTwice<f,1,27648,6912,0>::filter0 start = 13284,end = 40990,total = 27706
TransposeScalarBHWC2BCHW<f,1,24,96,3> start = 34074,end = 42044,total = 7970
TransposeScalarBHWC2BCHW<f,1,24,96,3> start = 40993,end = 48963,total = 7970
ConcatFloatStreamWithStall<f,3,2304,2304,4608> start = 13287,end = 52570,total = 39283
ConcatFloatStreamWithStall<f,3,2304,2304,4608> start = 13284,end = 66400,total = 53116
ConcatFloatStreamWithStall<f,3,4608,4608,9216> start = 13289,end = 84842,total = 71553
QuantizeLinearFmulStream<h,288,96,96> start = 13287,end = 87733,total = 74446
Pad2DStreamInt8<h,3,96,96,96,0,1,0,16> start = 13289,end = 87829,total = 74540
SplitFilterInt8PktStream<h,3,10864,3696,112>::filter3 start = 13299,end = 87865,total = 74566
QLinearConvHx4PktStream<h,h,33,112,48,48,2,2,1,3,8,3,3,1> start = 13356,end = 110718,total = 97362
ConcatInt8Stream<h,8,768,768,1536> start = 13297,end = 117345,total = 104048
QLinearConvHx4PktStream<h,h,33,112,48,48,2,2,1,3,8,3,3,1> start = 13357,end = 117343,total = 103986
ConcatInt8Stream<h,8,1536,768,2304> start = 13299,end = 123612,total = 110313
QLinearConvHx4PktStream<h,h,33,112,48,48,2,2,1,3,8,3,3,1> start = 13359,end = 123609,total = 110250
Pad2DStreamInt8<h,8,48,48,48,1,1,1,15> start = 13302,end = 123688,total = 110386
SplitFilterInt8PktStream<h,8,3200,1664,128>::filter2 start = 13306,end = 123720,total = 110414
QLinearConvHx4PktStream<h,h,26,64,48,48,1,1,1,8,8,3,3,8> start = 13360,end = 127510,total = 114150
QLinearConvHx4PktStream<h,h,26,64,48,48,1,1,1,8,8,3,3,8> start = 13361,end = 131010,total = 117649
ConcatInt8Stream<h,8,1152,1152,2304> start = 13301,end = 131020,total = 117719
SplitFilterInt8PktStream<h,8,2304,1152,0>::filter2 start = 13312,end = 131260,total = 117948
QLinearConv1x1PktStream<h,h,24,48,48,48,1,1,1,8,16,1,1,1> start = 13366,end = 164621,total = 151255
QLinearConv1x1PktStream<h,h,24,48,48,48,1,1,1,8,16,1,1,1> start = 13367,end = 165052,total = 151685
ConcatInt8Stream<h,16,1152,1152,2304> start = 13307,end = 165068,total = 151761
Pad2DStreamInt8<h,16,48,48,48,0,1,0,16> start = 13313,end = 165502,total = 152189
SplitFilterInt8PktStream<h,16,3136,832,64>::filter4 start = 13319,end = 165531,total = 152212
QLinearConvHx4PktStream<h,h,13,64,24,32,2,2,1,16,16,3,3,16> start = 13369,end = 176437,total = 163068
QLinearConvHx4PktStream<h,h,13,64,24,32,2,2,1,16,16,3,3,16> start = 13370,end = 177312,total = 163942
ConcatInt8Stream<h,16,192,192,384> start = 13310,end = 177319,total = 164009
QLinearConvHx4PktStream<h,h,13,64,24,32,2,2,1,16,16,3,3,16> start = 13372,end = 177707,total = 164335
QLinearConvHx4PktStream<h,h,13,64,24,32,2,2,1,16,16,3,3,16> start = 13373,end = 178067,total = 164694
ConcatInt8Stream<h,16,192,192,384> start = 13313,end = 178074,total = 164761
ConcatInt8Stream<h,16,384,384,768> start = 13315,end = 178084,total = 164769
Pad2DStreamInt8<h,16,24,24,32,0,0,0,8> start = 13311,end = 178126,total = 164815
QLinearConv1x1Stream<h,h,24,32,24,32,1,1,1,16,32,1,1,1> start = 178147,end = 241262,total = 63115
Pad2DStreamInt8<h,32,24,24,32,1,1,1,7> start = 13306,end = 241327,total = 228021
SplitFilterInt8PktStream<h,32,832,448,64>::filter2 start = 13310,end = 241351,total = 228041
QLinearConvHx4PktStream<h,h,14,32,24,32,1,1,1,32,32,3,3,32> start = 13369,end = 251344,total = 237975
QLinearConvHx4PktStream<h,h,14,32,24,32,1,1,1,32,32,3,3,32> start = 13370,end = 252131,total = 238761
ConcatInt8Stream<h,32,384,384,768> start = 13310,end = 252141,total = 238831
Pad2DStreamInt8<h,32,24,24,32,0,0,0,8> start = 13313,end = 260793,total = 247480
SplitFilterInt8PktStream<h,32,768,384,0>::filter2 start = 13317,end = 260818,total = 247501
QLinearConv1x1PktStream<h,h,12,32,24,32,1,1,1,32,32,1,1,1> start = 13375,end = 312033,total = 298658
QLinearConv1x1PktStream<h,h,12,32,24,32,1,1,1,32,32,1,1,1> start = 13376,end = 312306,total = 298930
ConcatInt8Stream<h,32,384,384,768> start = 13316,end = 312316,total = 299000
Pad2DStreamInt8<h,32,24,24,32,0,1,0,8> start = 13319,end = 312397,total = 299078
SplitFilterInt8PktStream<h,32,800,416,32>::filter2 start = 13323,end = 312423,total = 299100
QLinearConvHx4PktStream<h,h,13,32,12,16,2,2,1,32,32,3,3,32> start = 13379,end = 324874,total = 311495
QLinearConvHx4PktStream<h,h,13,32,12,16,2,2,1,32,32,3,3,32> start = 13380,end = 325128,total = 311748
ConcatInt8Stream<h,32,96,96,192> start = 13319,end = 325135,total = 311816
Pad2DStreamInt8<h,32,12,12,16,0,0,0,4> start = 13320,end = 325184,total = 311864
QLinearConv1x1Stream<h,h,12,16,12,16,1,1,1,32,64,1,1,1> start = 325203,end = 384257,total = 59054
Pad2DStreamInt8<h,64,12,12,16,1,1,1,3> start = 13317,end = 384316,total = 370999
QLinearConvHx4Stream<h,h,14,16,12,16,1,1,1,64,64,3,3,64> start = 384332,end = 398273,total = 13941
Pad2DStreamInt8<h,64,12,12,16,0,0,0,4> start = 13291,end = 402987,total = 389696
QLinearConv1x1Stream<h,h,12,16,12,16,1,1,1,64,64,1,1,1> start = 403007,end = 500418,total = 97411
Pad2DStreamInt8<h,64,12,12,16,0,1,0,4> start = 13276,end = 500487,total = 487211
QLinearConvHx4Stream<h,h,13,16,6,16,2,2,1,64,64,3,3,64> start = 500508,end = 511367,total = 10859
Pad2DStreamInt8<h,64,6,6,16,0,0,0,10> start = 13278,end = 511414,total = 498136
QLinearConv1x1Stream<h,h,6,16,6,16,1,1,1,64,128,1,1,1> start = 511434,end = 614050,total = 102616
Pad2DStreamInt8<h,128,6,6,16,1,1,1,9> start = 13244,end = 614126,total = 600882
QLinearConvHx4Stream<h,h,8,16,6,16,1,1,1,128,128,3,3,128> start = 614142,end = 628915,total = 14773
Pad2DStreamInt8<h,128,6,6,16,0,0,0,10> start = 13237,end = 634451,total = 621214
QLinearConv1x1Stream<h,h,6,16,6,16,1,1,1,128,128,1,1,1> start = 634471,end = 816789,total = 182318
Pad2DStreamInt8<h,128,6,6,16,1,1,1,9> start = 13234,end = 816848,total = 803614
QLinearConvHx4Stream<h,h,8,16,6,16,1,1,1,128,128,3,3,128> start = 816864,end = 831637,total = 14773
Pad2DStreamInt8<h,128,6,6,16,0,0,0,10> start = 13236,end = 837173,total = 823937
QLinearConv1x1Stream<h,h,6,16,6,16,1,1,1,128,128,1,1,1> start = 837193,end = 1019511,total = 182318
Pad2DStreamInt8<h,128,6,6,16,1,1,1,9> start = 13230,end = 1019573,total = 1006343
QLinearConvHx4Stream<h,h,8,16,6,16,1,1,1,128,128,3,3,128> start = 1019589,end = 1034362,total = 14773
Pad2DStreamInt8<h,128,6,6,16,0,0,0,10> start = 13228,end = 1039896,total = 1026668
QLinearConv1x1Stream<h,h,6,16,6,16,1,1,1,128,128,1,1,1> start = 1039916,end = 1222234,total = 182318
Pad2DStreamInt8<h,128,6,6,16,1,1,1,9> start = 13226,end = 1222294,total = 1209068
QLinearConvHx4Stream<h,h,8,16,6,16,1,1,1,128,128,3,3,128> start = 1222310,end = 1237083,total = 14773
Pad2DStreamInt8<h,128,6,6,16,0,0,0,10> start = 13228,end = 1242619,total = 1229391
QLinearConv1x1Stream<h,h,6,16,6,16,1,1,1,128,128,1,1,1> start = 1242639,end = 1424957,total = 182318
Pad2DStreamInt8<h,128,6,6,16,1,1,1,9> start = 13253,end = 1425028,total = 1411775
QLinearConvHx4Stream<h,h,8,16,6,16,1,1,1,128,128,3,3,128> start = 1425044,end = 1439817,total = 14773
Pad2DStreamInt8<h,128,6,6,16,0,0,0,10> start = 13258,end = 1445353,total = 1432095
QLinearConv1x1Stream<h,h,6,16,6,16,1,1,1,128,128,1,1,1> start = 1445373,end = 1627691,total = 182318
Pad2DStreamInt8<h,128,6,6,16,0,1,0,10> start = 13262,end = 1627756,total = 1614494
QLinearConvHx4Stream<h,h,7,16,3,16,2,2,1,128,128,3,3,128> start = 1627777,end = 1639445,total = 11668
Pad2DStreamInt8<h,128,3,3,16,0,0,0,13> start = 13266,end = 1639664,total = 1626398
QLinearConv1x1Stream<h,h,3,16,3,16,1,1,1,128,256,1,1,1> start = 1639684,end = 1838385,total = 198701
Pad2DStreamInt8<h,256,3,3,16,1,1,1,12> start = 13253,end = 1838449,total = 1825196
SplitFilterInt8Stream<h,256,80,48,32>::filter2 start = 13251,end = 1838459,total = 1825208
SplitFilterInt8StreamTwice<h,256,80,48,32>::filter0 start = 13246,end = 1838463,total = 1825217
QLinearConvHx4Stream<h,h,3,16,3,16,1,1,1,256,256,3,3,256> start = 1838434,end = 1843869,total = 5435
QLinearConvHx4Stream<h,h,3,16,3,16,1,1,1,256,256,3,3,256> start = 1838458,end = 1843893,total = 5435
ConcatInt8Stream<h,256,16,16,32> start = 13246,end = 1843904,total = 1830658
QLinearConvHx4Stream<h,h,3,16,3,16,1,1,1,256,256,3,3,256> start = 1838475,end = 1843910,total = 5435
ConcatInt8Stream<h,256,32,16,48> start = 13244,end = 1843922,total = 1830678
Pad2DStreamInt8<h,256,3,3,16,0,0,0,13> start = 13242,end = 1862091,total = 1848849
QLinearConv1x1Stream<h,h,3,16,3,16,1,1,1,256,256,1,1,1> start = 1862112,end = 2232079,total = 369967
QLinearAvgpoolScalarBCHW<3,16,1,1,1,256,3,3> start = 2232091,end = 2269741,total = 37650
QgemmStream<h,h,1,256,16> start = 13305,end = 2269788,total = 2256483
Pad2DStreamInt8<h,1,1,2,16,0,0,0,14> start = 13269,end = 2269844,total = 2256575
QLinearSoftmaxSingleaxis<h, 1,2,16> start = 2269856,end = 2270028,total = 172
DequantizeLinear<h,1,16,4> start = 2270046,end = 2270102,total = 56

Checking 1/14: k000transpose_goldenout_shape1x3x96x96.txt                                       against k000transpose_goldenout_shape1x3x96x96.txt              TEST: OK!
Checking 2/14: k001quantizelinear_goldenout_shape1x3x96x96.txt                                  against k001quantizelinear_goldenout_shape1x3x96x96.txt         TEST: OK!
Checking 3/14: k002qlinearconv_goldenout_shape1x8x48x48.txt                                     against k002qlinearconv_goldenout_shape1x8x48x48.txt            TEST: FAILED! Only 18430/18432 passed.
Max absolute difference: 1.0
Max relative difference: 0.027777777777777776
Checking 4/14: k003qlinearconv_goldenout_shape1x8x48x48.txt                                     against k003qlinearconv_goldenout_shape1x8x48x48.txt            TEST: FAILED! Only 18424/18432 passed.
Max absolute difference: 1.0
Max relative difference: 0.041666666666666664
Checking 5/14: k004qlinearconv_goldenout_shape1x16x48x48.txt                                    against k004qlinearconv_goldenout_shape1x16x48x48.txt           TEST: FAILED! Only 36845/36864 passed.
Max absolute difference: 1.0
Max relative difference: 0.25
Checking 6/14: k006qlinearconv_goldenout_shape1x32x24x24.txt                                    against k006qlinearconv_goldenout_shape1x32x24x24.txt           TEST: FAILED! Only 2229/18432 passed.
Max absolute difference: 254.0
Max relative difference: 254.0
Checking 7/14: k007qlinearconv_goldenout_shape1x32x24x24.txt                                    against k007qlinearconv_goldenout_shape1x32x24x24.txt           TEST: FAILED! Only 4591/18432 passed.
Max absolute difference: 254.0
Max relative difference: 254.0
Checking 8/14: k008qlinearconv_goldenout_shape1x32x24x24.txt                                    against k008qlinearconv_goldenout_shape1x32x24x24.txt           TEST: FAILED! Only 3311/18432 passed.
Max absolute difference: 254.0
Max relative difference: 254.0
Checking 9/14: k026qlinearconv_goldenout_shape1x256x3x3.txt                                     against k026qlinearconv_goldenout_shape1x256x3x3.txt            TEST: FAILED! Only 2118/2304 passed.
Max absolute difference: 42.0
Max relative difference: 15.0
Checking 10/14: k028qlinearconv_goldenout_shape1x256x3x3.txt                                     against k028qlinearconv_goldenout_shape1x256x3x3.txt           TEST: FAILED! Only 2189/2304 passed.
Max absolute difference: 55.0
Max relative difference: 10.0
Checking 11/14: k029qlinearpool_goldenout_shape1x256x1x1.txt                                     against k029qlinearpool_goldenout_shape1x256x1x1.txt           TEST: FAILED! Only 236/256 passed.
Max absolute difference: 22.0
Max relative difference: 21.0
Checking 12/14: k031qgemm_goldenout_shape1x2.txt                                                 against k031qgemm_goldenout_shape1x2.txt                       TEST: FAILED! Only 0/2 passed.
Max absolute difference: 27.0
Max relative difference: 0.26262626262626265
Checking 13/14: k032qlinearsoftmax_goldenout_shape1x2.txt                                        against k032qlinearsoftmax_goldenout_shape1x2.txt              TEST: FAILED! Only 0/2 passed.
Max absolute difference: 121.0
Max relative difference: 5.260869565217392
Checking 14/14: k033dequantizeLinear_goldenout_shape1x2.txt                                      against k033dequantizeLinear_goldenout_shape1x2.txt            TEST: FAILED! Only 0/2 passed.
Max absolute difference: 0.47265625
Max relative difference: 5.260869565217392
```