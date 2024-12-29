import numpy as np
import torch

from torchvision.models import resnet50, ResNet50_Weights

ONNX_PATH = "../models/imagenet1k_resnet50.onnx"

# Load model
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)
model.eval()

# Load data
data = np.load("../data/imagenet1k/X_test.npy")
data = torch.from_numpy(data)
labels = np.load("../data/imagenet1k/Y_test.npy")

# Convert to onnx and save
torch.onnx.export(
    model,                            # model being run
    data[[0]],                        # model input (or a tuple for multiple inputs)
    ONNX_PATH,                        # where to save the model
    export_params=True,               # store the trained parameter weights in model file
    do_constant_folding=True,         # whether to execute constant folding for optimization
    input_names = ["input"],          # model's input names
    output_names = ["output"],        # model's output names
    dynamic_axes={"input" : {0 : "batch_size"},    # variable length axes
                  "output" : {0 : "batch_size"}})
print("Converted to onnx")

# Validation
result = []
for i in range(1000):
  prediction = model(data[[i]]).squeeze(0).softmax(0)
  class_id = prediction.argmax().item()
  score = prediction[class_id].item()
  category_name = weights.meta["categories"][class_id]
  print(f"{category_name}: {100 * score:.1f}%")

  result.append(class_id)

result = np.array(result)
print(f"Top 1: {(labels == result).sum() / labels.size}")
