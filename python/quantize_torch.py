import torch
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from model import QuantizedLenet


if __name__ == "__main__":
  DATA_PATH = "../data"
  PKL_PATH = "../models/lenet_mnist.pkl"
  QPKL_PATH = "../models/lenet_mnist_quantized.pkl"
  QONNX_PATH = "../models/lenet_mnist_quantized.onnx"

  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Data
  dataset = mnist.MNIST(root=DATA_PATH, train=False, transform=ToTensor(), download=True)
  loader = DataLoader(dataset, batch_size=1)
  data, label = next(iter(loader))
  input_fp32 = data[0].unsqueeze(0)

  # create a model instance
  model_fp32 = torch.load(PKL_PATH).to(device)
  model_fp32.eval() # eval mode required for static quantization logic to work
  
  # global qconfig: type of observers to attach
  # 'x86' for server inference and 'qnnpack' for mobile inference
  # Other configs (e.g. symmetric / asymmetric quantization, MinMax, L2Norm calibration techniques can be specified
  # Note: the old 'fbgemm' is still available but 'x86' is the recommended default
  model_fp32.qconfig = torch.ao.quantization.get_default_qconfig("x86")
  
  # Fuse the activations to preceding layers, where applicable.
  # This needs to be done manually depending on the model architecture.
  # Common fusions include `conv + relu` and `conv + batchnorm + relu`
  # model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [['conv', 'relu']])
  
  # Inserts observers to observe activation tensors during calibration
  model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)

  # calibrate the prepared model to determine quantization parameters for activations
  # in a real world setting, the calibration would be done with a representative dataset
  model_fp32_prepared(input_fp32)

  # Convert the observed model to a quantized model. This does several things:
  # quantizes the weights, computes and stores the scale and bias value to be
  # used with each activation tensor, and replaces key operators with quantized
  # implementations.
  model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
  torch.save(model_int8, QPKL_PATH)

  # run the model, relevant calculations will happen in int8
  res = model_int8(input_fp32)
  
  torch.onnx.export(
    model_int8,                       # model being run
    input_fp32,                       # model input (or a tuple for multiple inputs)
    QONNX_PATH,                       # where to save the model
    export_params=True,               # store the trained parameter weights in model file
    do_constant_folding=True,         # whether to execute constant folding for optimization
    input_names = ["input"],          # model's input names
    output_names = ["output"],        # model's output names
    dynamic_axes={"input" : {0 : "batch_size"},    # variable length axes
                  "output" : {0 : "batch_size"}})
  print("Converted to onnx")
  # torch.onnx.errors.SymbolicValueError: ONNX symbolic expected the output of `%127 : Tensor(*, *) = onnx::Reshape[allowzero=0](%y, %126), scope: model.QuantizeModel::/model.Lenet::model # /home/ruien/workspace/onnx2versal/python/model.py:28:0
