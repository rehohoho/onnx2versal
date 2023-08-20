## Profiling

### Tiny MLPerf Open Division (hls4ml models)

Following models used are trained from hls4ml-finn repositories. Quality targets are taken from [Hls4ml MLPerf Tiny paper](https://cds.cern.ch/record/2826586/files/2206.11791.pdf).

|       Use Case         | Dtype            | Latency (cycles or ns) | Throughput (samples/s) | Resource Utilization (Kernels/Buffers/Stream/PLIO/GMIO) | Accuracy (first 1k) | Quality Target | Model 
|:----------------------:|:----------------:|:----------------------:|:----------------------:|:---------------------------------------:|:-------------------:|:----------------:|:-------------------:|
|   Keyword Spotting   | fp32 <br/> uint8 | 35076    <br/> 3159    | 75369   <br/> 1157407  | 46/56/116/5/24   <br/> 48/51/83/7/0      | 84.8% (Top 1)       | 82.5% (Top 1)    | [MLP](https://github.com/hls4ml-finn-mlperftiny/tiny_results_v0.7/blob/main/open/hls4ml-finn/code/kws/KWS-W3A3/training/model/models.py)
|   Anomaly Detection  | fp32 <br/> uint8 | 3165     <br/> 1014    | 3205128 <br/> 7142857  | 44/58/128/7/0    <br/> 46/48/76/2/0      | 0.830 (AUC)         | 0.83  (AUC)      | [AutoEncoder](https://github.com/hls4ml-finn-mlperftiny/tiny_results_v0.7/blob/main/open/hls4ml-finn/code/ad/AD08/training/keras_model.py)
| Image Classification | fp32 <br/> uint8 | 739274   <br/> 174992  | 4324    <br/> 22258    | 62/68/125/9/7    <br/> 90/95/144/2/5     | 84.1% (Top 1)       | 83.5% (Top 1)    | [CNN](https://github.com/hls4ml-finn-mlperftiny/tiny_results_v0.7/blob/main/open/hls4ml-finn/code/ic/RN07/training/resnet_v1_eembc.py)

### MLPerf Closed Division (MLPerf pretrained)

Following models used are pretrained models *directly* from MLPerf Tiny Benchmark. Quality targets are taken from [MLPerf Tiny closed division benchmarks](https://github.com/mlcommons/tiny/tree/master/benchmark).

|       Use Case         | Dtype            | Latency (cycles or ns) | Throughput (samples/s) | Resource Utilization (Kernels/Buffers/Stream/PLIO/GMIO) | Accuracy (first 1k) | Quality Target<br>(Closed&#160;Division) | Model 
|:----------------------:|:----------------:|:----------------------:|:----------------------:|:---------------------------------------:|:-------------------:|:----------------:|:-------------------:|
|   Keyword Spotting    | fp32 <br/> uint8 | 839599   <br/> 402772  | 4109    <br/> 8080     | 100/110/161/2/9  <br/> 108/119/166/2/9   | 91.1% (Top 1)       | 90%   (Top 1)    | [DS-CNN](https://github.com/mlcommons/tiny/blob/master/benchmark/training/keyword_spotting/keras_model.py)
|   Visual Wake Words   | fp32 <br/> uint8 | -        <br/> 2249859 | -       <br/> 3384     | -                <br/> 101/103/189/15/27 | 82.5% (Top 1)       | 80%   (Top 1)    | [MobileNet](https://github.com/mlcommons/tiny/blob/master/benchmark/training/visual_wake_words/vww_model.py)
| Image Classification  | fp32 <br/> uint8 | 808574   <br/> 492622  | 3410    <br/> 7201     | 134/150/264/14/9 <br/> 116/127/214/16/9  | 86.8% (Top 1)       | 85%   (Top 1)    | [ResNet](https://github.com/mlcommons/tiny/blob/master/benchmark/training/image_classification/keras_model.py)
|   Anomaly Detection   | fp32 <br/> uint8 | 28453    <br/> 4625    | 102944  <br/> 968992   | 122/156/255/11/16<br/> 138/143/199/2/0   | 0.880 (AUC)         | 0.85  (AUC)      | [Deep AutoEncoder](https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/keras_model.py)


### Metrics
* Latency is calculated based on cycle count from cycle-accurate aiesimulator through AI Engine programming logging API, specifically `aie::tile::current().cycles()`. Obtained through aiesimulator logs.
* Throughput is calculated based on output bandwidth over multiple iterations. Obtained by running `throughput.py` on aiesimulator output files and assumes AI engine is clocked at 1GHz.
* Accuracy is calculated by comparing argmaxed probabilities or finding mean squared anomaly scores then computing area under curve. Obtained by running `get_accuracy.py` on x86simulator outputs over 1000 samples.


## Procedure

1) Use pretrained models from MLPerf Tiny Benchmark or hls4ml models from links in table above.
2) Do ops fusion.
3) Do post-training quantization using ONNXRuntime
4) Generate AI Engine graph
5) Run on AIEsimulator for performance metrics, x86simulator for accuracy metrics

### Experiment Links

The models, data and results for table above can be found below. Extract them in the root directory of this repo. <br/>
* [Models and data used](https://drive.google.com/file/d/1xTdA5xwMr3M7DTGJ3BQoC6lKG6iUJT2G/view?usp=sharing)
* [Simulator results](https://drive.google.com/file/d/1OQNMryfzPzbp40_CbXxMu3EwbuD6Rml_/view?usp=sharing)

### Replication

See requirements at README.

For float32 model's latency and throughput run aiesimulator
```
GRAPH=tiny_kws
python fuse_onnx.py ../models/${GRAPH}.onnx ../models/${GRAPH}.onnx
python generate.py ../models/${GRAPH}.onnx ../data/$GRAPH/X_test.npy -ndata 10

TARGET=hw_emu GRAPH=${GRAPH} make graph aiesim_profile                                # latency
TARGET=hw_emu DOUT=0 DLOG=0 GRAPH=${GRAPH} make graph clean_reports aiesim ITER_CNT=2 # throughput
python throughput.py reports_dir/$GRAPH/hw_emu/aiesimulator_output/k*
```

For int8 model's latency and throughput run aiesimulator
```
GRAPH=tiny_kws
python fuse_onnx.py ../models/${GRAPH}.onnx ../models/${GRAPH}.onnx
python -m onnxruntime.quantization.preprocess --input ../models/${GRAPH}.onnx --output ../models/${GRAPH}_infer.onnx
python quantize_onnx.py ../models/${GRAPH}_infer.onnx ../models/${GRAPH}_int8.onnx ../data/$GRAPH/X_test.npy
python generate.py ../models/${GRAPH}_int8.onnx ../data/$GRAPH/X_test.npy

TARGET=hw_emu GRAPH=${GRAPH}_int8 make graph aiesim_profile                           # latency
TARGET=hw_emu DOUT=0 DLOG=0 GRAPH=${GRAPH} make graph clean_reports aiesim ITER_CNT=2 # throughput
python throughput.py reports_dir/$GRAPH/hw_emu/aiesimulator_output/k*
```

For accuracy, run x86simulator
```
TARGET=sw_emu EXTIO=0 DOUT=0 DLOG=0 GRAPH=${GRAPH} make graph clean_reports aiesim ITER_CNT=1000
```

### Result outputs
This codebase outputs start and ending cycles for each kernel used when running aiesimulator. At the end of each simulation, it does a `np.allclose` check against golden values from running the ONNX model with ONNXRuntime with absolute tolerance 1e-05, relative tolerance 1e-03. If `absolute(a - b) <= (atol + rtol * absolute(b))` is element-wise True, the check returns OK.
