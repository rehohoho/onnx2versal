from typing import List, Mapping
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import numpy_helper


class OpParser:
  include_file: str

  def __init__(self, name: str):
    self.name = name

  def save_output(self, output: np.ndarray):
    np.savetxt(f"../data/{self.name}_goldenout.txt", output.reshape(-1, 2))
  
  def get_include_line(self) -> str:
    return f'#include "{self.include_file}"'
  
  def get_arg_line(self) -> str:
    return f""
  
  def get_initlist_line(self) -> str:
    return f""
  
  def get_weight_line(self) -> str:
    return f""
  
  def get_callarg_line(self) -> str:
    return f""
