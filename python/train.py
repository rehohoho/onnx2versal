import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
import torch.onnx
import onnxruntime

from model import Lenet, QuantizedLenet
from op_parsers import save_tensor


def get_n_data(dataset: torch.utils.data.Dataset,
               data_count: int):
  data = None
  print(f"Generating MNIST txt for {data_count} data points")
  for x, y in dataset:
    if data is None:
      data = x.unsqueeze(0)
    else:
      data = torch.concat((data, x.unsqueeze(0)))
    if data.shape[0] >= data_count:
      break
  return data


if __name__ == "__main__":
  BATCH_SIZE = 256
  ALL_EPOCH = 1
  OUTPUT_DATACOUNT = 100
  DATA_PATH = "../data"
  PKL_PATH = "../models/lenet_mnist.pkl"
  ONNX_PATH = "../models/lenet_mnist.onnx"

  # Setup data, model, optimizer, loss_fn
  device = "cuda" if torch.cuda.is_available() else "cpu"
  train_dataset = mnist.MNIST(root=DATA_PATH, train=True, transform=ToTensor(), download=True)
  test_dataset = mnist.MNIST(root=DATA_PATH, train=False, transform=ToTensor(), download=True)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
  model = Lenet().to(device)
  sgd = SGD(model.parameters(), lr=1e-1)
  loss_fn = CrossEntropyLoss()
  prev_acc = 0

  # Train
  for current_epoch in range(ALL_EPOCH):
    model.train()
    for idx, (train_x, train_label) in enumerate(train_loader):
      train_x = train_x.to(device)
      train_label = train_label.to(device)
      sgd.zero_grad()
      predict_y = model(train_x.float())
      loss = loss_fn(predict_y, train_label.long())
      loss.backward()
      sgd.step()

    all_correct_num = 0
    all_sample_num = 0
    model.eval()
    
    for idx, (test_x, test_label) in enumerate(test_loader):
      test_x = test_x.to(device)
      test_label = test_label.to(device)
      predict_y = model(test_x.float()).detach()
      predict_y =torch.argmax(predict_y, dim=-1)
      current_correct_num = predict_y == test_label
      all_correct_num += np.sum(current_correct_num.to("cpu").numpy(), axis=-1)
      all_sample_num += current_correct_num.shape[0]

    acc = all_correct_num / all_sample_num
    print(f"accuracy: {acc:.3f}", flush=True)
    if not os.path.isdir(os.path.dirname(PKL_PATH)):
        os.mkdir(os.path.dirname(PKL_PATH))
    torch.save(model, PKL_PATH)
    
    if np.abs(acc - prev_acc) < 1e-4:
        break  
    prev_acc = acc
  
  print("Model finished training")

  # ONNX export
  torch.onnx.export(
    model,                            # model being run
    test_x,                           # model input (or a tuple for multiple inputs)
    ONNX_PATH,                        # where to save the model
    export_params=True,               # store the trained parameter weights in model file
    do_constant_folding=True,         # whether to execute constant folding for optimization
    input_names = ["input"],          # model's input names
    output_names = ["output"],        # model's output names
    dynamic_axes={"input" : {0 : "batch_size"},    # variable length axes
                  "output" : {0 : "batch_size"}})
  print("Converted to onnx")

  # Compare ONNX Runtime and PyTorch results
  data, label = next(iter(test_loader))
  input_tensor = data[0].unsqueeze(0)
  torch_out = model.forward(input_tensor)

  ort_session = onnxruntime.InferenceSession(ONNX_PATH)
  ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
  ort_outs = ort_session.run(None, ort_inputs)

  np.testing.assert_allclose(torch_out.detach().numpy(), ort_outs[-1], rtol=1e-03, atol=1e-05)
  print("Exported model has been tested with ONNXRuntime, and the result looks good!")

  # Save data for verification
  # Single sample for intermediate outputs, multiple for end to end
  e2e_data = get_n_data(test_dataset, OUTPUT_DATACOUNT).numpy()
  save_tensor(f"{DATA_PATH}/{ort_session.get_inputs()[0].name}_host.txt", e2e_data)
  save_tensor(f"{DATA_PATH}/{ort_session.get_inputs()[0].name}.txt", e2e_data[0])
