## Profiling

(on aiesim)

### hls4ml_jettag
```
# Run
python generate.py ../models/hls4ml_jettag.onnx ../data/MNIST/X_test.npy
TARGET=hw_emu GRAPH=hls4ml_jettag make graph aiesim_profile
```

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
# Run
python -m onnxruntime.quantization.preprocess --input ../models/hls4ml_jettag.onnx --output ../models/hls4ml_jettag_infer.onnx
python quantize_onnx.py ../models/hls4ml_jettag_infer.onnx ../models/hls4ml_jettag_int8.onnx ../data/MNIST/X_test.npy
python generate.py ../models/hls4ml_jettag_int8.onnx ../data/MNIST/X_test.npy
TARGET=hw_emu GRAPH=hls4ml_jettag_int8 make graph aiesim_profile
```

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
# Run
python generate.py ../models/lenet_mnist.onnx ../data/MNIST/X_test.npy
TARGET=hw_emu GRAPH=lenet_mnist make graph aiesim_profile
```

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
Running ConcatVector<8,1,16,120>::filter8
start = 55936,end = 56062,total = 126
Running GemmReluMKKN<1,120,32>
Running GemmReluMKKN<1,120,32>
Running GemmReluMKKN<1,120,32>
start = 56382,end = 57270,total = 888
start = 56386,end = 57274,total = 888
start = 56390,end = 57278,total = 888
Running ConcatVector<3,1,32,84>::filter3
start = 57426,end = 57499,total = 73
Running GemmReluMKKN<1,84,48>
start = 57771,end = 58744,total = 973
Running ConcatScalar<1,1,48,10>::filter1
start = 58888,end = 59766,total = 878
```
total 48037 cycles, note GemmReluMKKN runs in parallel

### lenet_mnist_int8
```
# Run
python -m onnxruntime.quantization.preprocess --input ../models/lenet_mnist.onnx --output ../models/lenet_mnist_infer.onnx
python quantize_onnx.py ../models/lenet_mnist_infer.onnx ../models/lenet_mnist_int8.onnx ../data/MNIST/X_test.npy
python generate.py ../models/lenet_mnist_int8.onnx ../data/MNIST/X_test.npy
TARGET=hw_emu GRAPH=lenet_mnist_int8 make graph aiesim_profile
```

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