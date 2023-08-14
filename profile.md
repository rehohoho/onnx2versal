## Profiling

|       Use Case         | Dtype            | Latency (cycles or ns) | Throughput (instances/s) | Resource Utilization (Kernels/Buffers/Stream/PLIO/GMIO) | Accuracy (first 1k) | Quality Target<br>(Closed&#160;Division) | Model 
|:----------------------:|:----------------:|:----------------------:|:------------------------:|:---------------------------------------:|:-------------------:|:----------------:|:-------------------:|
|   Keyword Spotting*    | fp32 <br/> uint8 | 839599   <br/> 402772  | 4109  <br/> 8080         | 100/110/161/2/9 <br/> 108/119/166/2/9   | 91.1% (Top 1)       | 90%   (Top 1)    | [DS-CNN](https://github.com/mlcommons/tiny/blob/master/benchmark/training/keyword_spotting/keras_model.py)
|   Visual Wake Words*   | fp32 <br/> uint8 | -        <br/> 2249859 | -     <br/> 3384         |                 <br/> 101/103/189/15/27 | 82.5% (Top 1)       | 80%   (Top 1)    | [MobileNet](https://github.com/mlcommons/tiny/blob/master/benchmark/training/visual_wake_words/vww_model.py)
| Image Classification*  | fp32 <br/> uint8 | 1028165  <br/> 763915  | 3410  <br/> 7201         | -               <br/> 41/49/112/12/9    | 86.8% (Top 1)       | 85%   (Top 1)    | [ResNet](https://github.com/mlcommons/tiny/blob/master/benchmark/training/image_classification/keras_model.py)
|   Anomaly Detection*   | fp32 <br/> uint8 | 25843356 <br/> 422940  | 608   <br/> 7603         | 55/97/208/20/2  <br/> 147/159/202/0/2   | 0.880 (AUC)         | 0.85  (AUC)      | [Deep AutoEncoder](https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/keras_model.py)
|   Keyword Spotting**   | fp32 <br/> uint8 | 35076    <br/> 3159    | 75369 <br/> 1157407      | 46/56/116/5/24  <br/> 48/50/81/0/7      | 84.8% (Top 1)       | 82.5% (Top 1)    | [MLP](https://github.com/hls4ml-finn-mlperftiny/tiny_results_v0.7/blob/main/open/hls4ml-finn/code/kws/KWS-W3A3/training/model/models.py)
|   Anomaly Detection**  | fp32 <br/> uint8 | 442442   <br/> 44224   | 11081 <br/> 35592        | 11/18/83/0/8    <br/> 48/50/84/0/10     | 0.832 (AUC)         | 0.83  (AUC)      | [AutoEncoder](https://github.com/hls4ml-finn-mlperftiny/AnomalyDetection/blob/main/keras_model.py)
| Image Classification** | fp32 <br/> uint8 | -        <br/> 308461  | -     <br/> 7355         | -               <br/>                   | 84.1% (Top 1)       | 83.5% (Top 1)    | [CNN](https://github.com/hls4ml-finn-mlperftiny/tiny_results_v0.7/blob/main/open/hls4ml-finn/code/ic/RN07/training/resnet_v1_eembc.py)
\*Models used are pretrained models *directly* from MLPerf Tiny Benchmark.
\*\*Models used are trained from hls4ml-finn repositories.

* Latency is calculated based on AI Engine programming logging API, specifically `aie::tile::current().cycles()`. Obtained through aiesimulator outputs.
* Throughput is calculated based on output bandwidth over multiple iterations. Obtained by running `throughput.py` on aiesimulator output files.
* Accuracy is calculated by comparing argmaxed probabilities or finding mean squared anomaly scores then computing area under curve. Obtained by running `get_accuracy.py` on x86simulator outputs.


## Procedure

1) Use pretrained models from MLPerf Tiny Benchmark
2) Do post-training quantization using ONNXRuntime
3) Generate AI Engine graph
4) Run on AIEsimulator for performance metrics, x86simulator for accuracy metrics

The models, data and results for table above can be found below. Extract them in the root directory of this repo. <br/>
* Models and data used: `https://drive.google.com/file/d/1F55jfGdjst1MbBGwB1tkv3eFo9K0C-5O/view?usp=sharing`
* Simulator results: `https://drive.google.com/file/d/1FFEAAFhpRfnhBMxM4pMv-Tf2GHK1omXM/view?usp=sharing`


## Replication

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

## Result outputs
This codebase outputs start and ending cycles for each kernel used when running aiesimulator. At the end of each simulation, it does a `np.allclose` check against golden values from running the ONNX model with ONNXRuntime with absolute tolerance 1e-05, relative tolerance 1e-03. If `absolute(a - b) <= (atol + rtol * absolute(b))` is element-wise True, the check returns OK.
