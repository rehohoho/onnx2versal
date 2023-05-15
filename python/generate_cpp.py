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


class ArgmaxOp(OpParser):
  include_file: str = "graph_argmax.h"


class ConvOp(OpParser):
  include_file: str = "graph_conv.h"

  def register_params(self, 
                      tinput: np.ndarray, 
                      tweight: np.ndarray,
                      tbias: np.ndarray,
                      toutput: np.ndarray):
    inB, inC, inH, inW = tinput.shape
    wM, wC, wK1, wK2 = tweight.shape
    bM, = tbias.shape
    outB, outM, outH, outW = toutput.shape
    
    assert inH == inW and outH == outW and inC == wC and wM == bM and bM == outM \
      and inB == outB and wK1 == wK2

    self.INP_W = inH
    self.OUT_W = outH
    self.B = inB
    self.C = inC
    self.M = wM
    self.K = wK1
  
  def register_weights(self, 
                       weights: np.ndarray, 
                       bias: np.ndarray):
    """Expects MxCxKxK weights as per PyTorch
    Returns MxCxKxK' weights, with K' padded so K'%8=0
    """
    M, C, K, K = weights.shape
    k_pad = (8 - K%8) % 8
    if k_pad != 0:
      print(f"Padding Conv weights {M, C, K, K} to {M, C, K, K+k_pad}")
      weights = np.pad(weights, ((0,0),(0,0),(0,0),(0,k_pad)), "constant", constant_values=0)
    self.weights = weights
    self.bias = bias
  
  def get_kernel_line(self) -> str:
    return f"ConvReluGraph<Conv5x5on8ReluBCHW,{self.INP_W},{self.OUT_W},{self.B},{self.C},{self.M},{self.K}> {self.name};"

  def get_arg_line(self) -> str:
    return f"std::vector<float> {self.name}_w,\nstd::vector<float> {self.name}_b"
  
  def get_initlist_line(self) -> str:
    return f"{self.name}({self.name}_w, {self.name}_b)"
  
  def get_weight_line(self) -> str:
    wstring = str(self.weights.flatten().tolist())[1:-1]
    bstring = str(self.bias.flatten().tolist())[1:-1]
    return f"std::vector<float> {self.name}_w {{{wstring}}};\n" + \
      f"std::vector<float> {self.name}_b {{{bstring}}};"
  
  def get_callarg_line(self) -> str:
    return f"{self.name}_w, {self.name}_b"
  

class GemmOp(OpParser):
  include_file: str = "graph_gemm.h"

  def register_params(self, 
                      tinput: np.ndarray, 
                      tweight: np.ndarray,
                      tbias: np.ndarray,
                      toutput: np.ndarray):
    inM, inK = tinput.shape
    wN, wK = tweight.shape
    bN, = tbias.shape
    outM, outN = toutput.shape
    
    assert inM == outM and inK == wK and wN == bN and bN == outN

    self.M = inM
    self.K = inK
    self.N = wN
  
  def register_weights(self, 
                       weights: np.ndarray,
                       bias: np.ndarray):
    """Expects NxK weights as per PyTorch
    Returns KxN weights, with N padded so N%4=0
    """
    weights = weights.transpose(1,0)
    K, N = weights.shape
    n_pad = (4 - N%4) % 4
    if n_pad != 0:
      print(f"Padding Gemm weights {K, N} to {K, N+n_pad}")
      weights = np.pad(weights, ((0,0),(0,n_pad)), "constant", constant_values=0)
    self.weights = weights
    self.bias = bias
  
  def get_kernel_type(self) -> str:
    return f"GemmReluMkknChunkGraph<GemmReluMKKN,ConcatVector,MAX_FLOAT_PARAMS/{self.K}/4*4,{self.M},{self.K},{self.N}>"
  
  def get_kernel_line(self) -> str:
    return f"{self.get_kernel_type()} {self.name};"

  def get_arg_line(self) -> str:
    return f"std::vector<float> {self.name}_w,\nstd::vector<float> {self.name}_b"
  
  def get_initlist_line(self) -> str:
    return f"{self.name}({self.name}_w, {self.name}_b)"
  
  def get_weight_line(self) -> str:
    wstring = str(self.weights.flatten().tolist())[1:-1]
    bstring = str(self.bias.flatten().tolist())[1:-1]
    return f"std::vector<float> {self.name}_w {{{wstring}}};\n" + \
      f"std::vector<float> {self.name}_b {{{bstring}}};"
  
  def get_callarg_line(self) -> str:
    return f"{self.name}_w, {self.name}_b"


class PoolOp(OpParser):
  include_file: str = "graph_pool.h"
  
  def register_params(self, 
                      tinput: np.ndarray, 
                      toutput: np.ndarray):
    inB, inC, inH, inW = tinput.shape
    outB, outC, outH, outW = toutput.shape
    
    assert inH == inW and outH == outW and inC == outC and inB == outB

    self.INP_W = inH
    self.OUT_W = outH
    self.B = inB
    self.C = inC
  
  def get_kernel_line(self) -> str:
    return f"MaxpoolGraph<Maxpool2x2BCHW,{self.INP_W},{self.OUT_W},{self.B},{self.C}> {self.name};"
