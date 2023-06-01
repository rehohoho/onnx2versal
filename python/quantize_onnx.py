import argparse
import time

import numpy as np
import onnxruntime
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization import CalibrationDataReader


class NumpyDataReader(CalibrationDataReader):
  def __init__(self, 
               onnx_path: str,
               input_arrs: np.ndarray):
    self.enum_data = None
    self.data_list = [input_arrs[[i]] for i in range(input_arrs.shape[0])]

    session = onnxruntime.InferenceSession(onnx_path, None)
    self.input_name = session.get_inputs()[0].name
  
  def get_next(self):
    if self.enum_data is None:
      self.enum_data = iter(
        [{self.input_name: data} for data in self.data_list])
    return next(self.enum_data, None)

  def rewind(self):
    self.enum_data = None


def run_model(onnx_path: str, 
              input_arrs: np.ndarray):
  session = onnxruntime.InferenceSession(onnx_path)
  input_name = session.get_inputs()[0].name
  
  # Warm up
  _ = session.run([], {session.get_inputs()[0].name: input_arrs[[0]]})
  
  start = time.perf_counter()
  out = session.run([], {input_name: input_arrs})
  end = (time.perf_counter() - start) * 1000
  print(f"{onnx_path} took: {end:.2f}ms")

  return out


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Quantize ONNX model.')
  parser.add_argument("onnx",      nargs=1, help="required path to onnx file")
  parser.add_argument("qonnx",     nargs=1, help="required path to output quantized onnx file")
  parser.add_argument("input_npy", nargs=1, help="path to input data .npy, assume first dim is batch")
  parser.add_argument("-data", type=str, default="../data", help="path to data directory")
  parser.add_argument("-is_class", action="store_true", help="whether to argmax during run_model")
  args = parser.parse_args()
  args.onnx = args.onnx[0]
  args.qonnx = args.qonnx[0]
  args.input_npy = args.input_npy[0]
  
  Q_FORMAT = QuantFormat.QOperator # QDQ alternative: dequantize -> op -> quantize
  Q_PER_CHANNEL = False

  input_arrs = np.load(args.input_npy)
  data_reader = NumpyDataReader(onnx_path=args.onnx, input_arrs=input_arrs)

  # Calibrate and quantize model
  # Turn off model optimization during quantization
  quantize_static(
    args.onnx,
    args.qonnx,
    data_reader,
    quant_format=Q_FORMAT,
    per_channel=Q_PER_CHANNEL,
    weight_type=QuantType.QInt8,
    optimize_model=False,
  )
  print("Calibrated and quantized model saved.")

  # Assume single output
  out = run_model(onnx_path=args.onnx, input_arrs=input_arrs)[-1]
  qout = run_model(onnx_path=args.qonnx, input_arrs=input_arrs)[-1]

  # Output statistics
  close_count = np.isclose(out, qout, rtol=1e-03, atol=1e-05).sum()
  close_perc = 100*close_count/qout.size
  print(f"Matched elements: {close_count} / {qout.size} ({close_perc}%)")
  
  error = np.abs(out - qout)
  ref = np.abs(qout)
  nonzero = np.nonzero(ref)
  
  relerr = np.max(error[nonzero] / ref[nonzero])
  abserr = np.max(error[nonzero])
  print(f"Max absolute difference: {abserr}\nMax relative difference: {relerr}")

  # Output classification statistics
  print("\nRunning argmax on last dim...")
  out_cls = out.argmax(-1)
  qout_cls = qout.argmax(-1)
  match_count = (out_cls == qout_cls).sum()
  match_perc = 100*match_count/qout_cls.size
  print(f"Matched elements: {match_count} / {qout_cls.size} ({match_perc}%)")
