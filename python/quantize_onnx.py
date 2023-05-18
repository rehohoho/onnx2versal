import os
import argparse
import time

import numpy as np
import onnxruntime
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization import CalibrationDataReader
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from PIL import Image

from generate import get_n_data


class MnistDataReader(CalibrationDataReader):
  def __init__(self, 
               data_path: str, 
               data_count: int, 
               onnx_path: str):
    self.enum_data = None

    self.torch_dataset = mnist.MNIST(root=DATA_PATH, train=False, 
                                     transform=ToTensor(), download=True)
    self.data_list = [self.torch_dataset[i][0].unsqueeze(0).numpy() 
                      for i in range(data_count)]
    self.datasize = len(self.data_list)

    # Use inference session to get input shape.
    session = onnxruntime.InferenceSession(onnx_path, None)
    self.input_name = session.get_inputs()[0].name

  def get_next(self):
    if self.enum_data is None:
      self.enum_data = iter(
        [{self.input_name: data} for data in self.data_list]
      )
    return next(self.enum_data, None)

  def rewind(self):
    self.enum_data = None
    

def benchmark(onnx_path: str, input_data: np.ndarray):
  session = onnxruntime.InferenceSession(onnx_path)
  input_name = session.get_inputs()[0].name
  input_shape = [i if isinstance(i, int) else 1
                 for i in session.get_inputs()[0].shape]
  
  total = 0.0
  runs = 10

  # Warming up
  _ = session.run([], {session.get_inputs()[0].name: input_data})
  for i in range(runs):
    start = time.perf_counter()
    out = session.run([], {input_name: input_data})
    end = (time.perf_counter() - start) * 1000
    total += end
  total /= runs
  print(f"Avg: {total:.2f}ms over {runs} runs")
  return out


if __name__ == "__main__":
  """
  See https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
  cd ../models
  python -m onnxruntime.quantization.preprocess --input lenet_mnist.onnx --output lenet_mnist_infer.onnx
  python -m onnxruntime.quantization.preprocess --help
  cd ../python
  python quantize_onnx.py
  """
  
  DATA_COUNT = 100
  DATA_PATH = "../data"
  ONNX_PATH = "../models/lenet_mnist_infer.onnx"
  QONNX_PATH = "../models/lenet_mnist_int8.onnx"
  Q_FORMAT = QuantFormat.QOperator # QDQ alternative: dequantize -> op -> quantize
  Q_PER_CHANNEL = False

  data_reader = MnistDataReader(data_path=DATA_PATH, 
                                data_count=DATA_COUNT,
                                onnx_path=ONNX_PATH)

  # Calibrate and quantize model
  # Turn off model optimization during quantization
  quantize_static(
    ONNX_PATH,
    QONNX_PATH,
    data_reader,
    quant_format=Q_FORMAT,
    per_channel=Q_PER_CHANNEL,
    weight_type=QuantType.QInt8,
    optimize_model=False,
  )
  print("Calibrated and quantized model saved.")

  print("benchmarking fp32 model...")
  outs = benchmark(ONNX_PATH, data_reader.data_list[0])

  print("benchmarking int8 model...")
  qouts = benchmark(QONNX_PATH, data_reader.data_list[0])

  np.testing.assert_allclose(outs[-1], qouts[-1], rtol=1e-03, atol=1e-05)
