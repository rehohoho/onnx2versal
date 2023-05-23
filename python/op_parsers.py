from typing import List

import numpy as np

MAX_FLOAT_PARAMS = 16384//4
PLIO_WIDTH = 64 // 8 # in bytes
VECTOR_WORD_BOUNDARY = 16 # in bytes


def dtype_to_cstr(np_dtype: np.dtype):
  if np_dtype == "float32":
    return "float_t"
  elif np_dtype == "int32":
    return "int32_t"
  elif np_dtype == "int8":
    return "int8_t"
  else:
    raise NotImplementedError(f"Not implemented type {np_dtype}")


def save_tensor(output_path: str, 
                tensor: np.ndarray):
    n_lines = PLIO_WIDTH//tensor.dtype.itemsize
    if tensor.size > n_lines: 
      tensor = tensor.reshape(-1, n_lines)
    else: 
      tensor = tensor.reshape(-1)
    fmt = "%.9e"
    if "int" in str(tensor.dtype):
      fmt = "%d"
    np.savetxt(output_path, tensor, fmt=fmt)


def pad_lastdim(tensor: np.ndarray, 
                tensor_name: str, 
                N: int,
                value: int = 0):
  lastdim = tensor.shape[-1]
  pad_size = (N - lastdim%N) % N
  if pad_size != 0:
    print(f"Padding {tensor_name} {tensor.shape} to {*tensor.shape[:-1], lastdim+pad_size}")
    pad_arr = (*((0,0) for _ in range(tensor.ndim-1)),(0,pad_size))
    tensor = np.pad(tensor, pad_arr, "constant", constant_values=value)
  return tensor


class OpParser:
  include_file: str

  def __init__(self, name: str):
    self.name = name
    self.outname_2_tensors = {}
  
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
  
  def save_txt(self, data_path: str):
    for outname, outtensor in self.outname_2_tensors.items():
      save_tensor(f"{data_path}/{outname}", outtensor)
      print(f"Saved tensor of shape {outtensor.shape} into {outname}")


class ArgmaxOp(OpParser):
  include_file: str = "graph_argmax.h"


class ConvOp(OpParser):
  include_file: str = "graph_conv.h"

  def register_params(self, tensors: List[np.ndarray]):
    """Expects MxCxKxK weights as per PyTorch
    Returns MxCxKxK' weights, with K' padded so K'%8=0
    """
    assert len(tensors) == 4
    tin, tw, tbias, tout = tensors

    inB, inC, inH, inW = tin.shape
    wM, wC, wK1, wK2 = tw.shape
    bM, = tbias.shape
    outB, outM, outH, outW = tout.shape
    
    assert inH == inW and outH == outW and inC == wC and wM == bM and bM == outM \
      and inB == outB and wK1 == wK2

    self.INP_W, self.OUT_W, self.B, self.C, self.M, self.K = inH, outH, inB, inC, wM, wK1 # config
    self.dtype = tw.dtype
    self.out_size = tout.size
    self.tout = tout

    self.weights = pad_lastdim(tw, "Conv weights", 8) # heap
    self.bias = tbias

    self.outname_2_tensors[f"{self.name}_in.txt"] = tin # files
    self.outname_2_tensors[f"{self.name}_goldenout.txt"] = tout
  
  def get_kernel_line(self) -> str:
    graph = "ConvReluGraph"
    kernel = "ConvReluScalarBCHW"
    if self.OUT_W % 8 == 0 and self.K == 5:
      kernel = "Conv5x5on8ReluBCHW"
    
    return f"{graph}<{kernel},{self.INP_W},{self.OUT_W},{self.B},{self.C},{self.M},{self.K}> {self.name};"

  def get_arg_line(self) -> str:
    ctype = dtype_to_cstr(self.dtype)
    return f"std::vector<{ctype}> {self.name}_w,\nstd::vector<{ctype}> {self.name}_b"
  
  def get_initlist_line(self) -> str:
    return f"{self.name}({self.name}_w, {self.name}_b)"
  
  def get_weight_line(self) -> str:
    wstring = str(self.weights.flatten().tolist())[1:-1]
    bstring = str(self.bias.flatten().tolist())[1:-1]
    ctype = dtype_to_cstr(self.dtype)
    return f"std::vector<{ctype}> {self.name}_w {{{wstring}}};\n" + \
      f"std::vector<{ctype}> {self.name}_b {{{bstring}}};"
  
  def get_callarg_line(self) -> str:
    return f"{self.name}_w, {self.name}_b"
  

class GemmOp(OpParser):
  include_file: str = "graph_gemm.h"

  def register_params(self, tensors: List[np.ndarray]):
    """Expects NxK weights as per PyTorch
    Returns KxN weights, with N padded so N%4=0
    """
    assert len(tensors) == 4
    tin, tw, tbias, tout = tensors

    inM, inK = tin.shape
    wN, wK = tw.shape
    bN, = tbias.shape
    outM, outN = tout.shape
    
    assert inM == outM and inK == wK and wN == bN and bN == outN

    self.M, self.K, self.N = inM, inK, wN # config
    self.dtype = tw.dtype
    self.out_size = self.M*self.N
    self.tout = tout

    tw = tw.transpose(1,0) # heap
    self.weights = tw = pad_lastdim(tw, "Gemm weights", 4) 
    self.bias = tbias
  
    self.outname_2_tensors[f"{self.name}_in.txt"] = tin #files
    self.outname_2_tensors[f"{self.name}_goldenout.txt"] = tout

  def get_kernel_type(self) -> str:
    graph = "GemmReluMkknChunkGraph"
    kernel = "GemmReluScalarMKKN"
    concat_kernel = "ConcatScalar"
    chunkSize = int(MAX_FLOAT_PARAMS/self.K//4*4)
    if self.K % 2 == 0 and chunkSize % 4 == 0:
      kernel = "GemmReluMKKN"
    if chunkSize % 8 == 0 and self.N % 4 == 0:
      concat_kernel = "ConcatVector"
    
    return f"{graph}<{kernel},{concat_kernel},{chunkSize},{self.M},{self.K},{self.N}>"
  
  def get_kernel_line(self) -> str:
    return f"{self.get_kernel_type()} {self.name};"

  def get_arg_line(self) -> str:
    ctype = dtype_to_cstr(self.dtype)
    return f"std::vector<{ctype}> {self.name}_w,\nstd::vector<{ctype}> {self.name}_b"
  
  def get_initlist_line(self) -> str:
    return f"{self.name}({self.name}_w, {self.name}_b)"
  
  def get_weight_line(self) -> str:
    wstring = str(self.weights.flatten().tolist())[1:-1]
    bstring = str(self.bias.flatten().tolist())[1:-1]
    ctype = dtype_to_cstr(self.dtype)
    return f"std::vector<{ctype}> {self.name}_w {{{wstring}}};\n" + \
      f"std::vector<{ctype}> {self.name}_b {{{bstring}}};"
  
  def get_callarg_line(self) -> str:
    return f"{self.name}_w, {self.name}_b"


class PoolOp(OpParser):
  include_file: str = "graph_pool.h"
  
  def register_params(self, tensors: List[np.ndarray]):
    assert len(tensors) == 2
    tin, tout = tensors

    inB, inC, inH, inW = tin.shape
    outB, outC, outH, outW = tout.shape
    
    assert inH == inW and outH == outW and inC == outC and inB == outB

    self.INP_W, self.OUT_W, self.B, self.C = inH, outH, inB, inC #config
    self.dtype = tout.dtype
    self.out_size = tout.size
    self.tout = tout

    self.outname_2_tensors[f"{self.name}_in.txt"] = tin #files
    self.outname_2_tensors[f"{self.name}_goldenout.txt"] = tout
  
  def get_kernel_line(self) -> str:
    graph = "MaxpoolGraph"
    kernel = "MaxpoolScalarBCHW"
    if self.OUT_W % 4 == 0:
      kernel = "Maxpool2x2BCHW"
    return f"{graph}<{kernel},{self.INP_W},{self.OUT_W},{self.B},{self.C}> {self.name};"


class QuantizeLinearOp(OpParser):
  include_file: str = "graph_quantize_linear.h"

  def register_params(self, tensors: List[np.ndarray]):
    assert len(tensors) == 4
    tin, tscale, tzero, tout = tensors

    assert tin.size == tout.size and tscale.shape == () and tzero.shape == ()
    
    self.WINDOW_SIZE = tin.size # config
    self.dtype = tout.dtype
    self.out_size = tout.size

    self.scale = tscale.item() # heap
    self.zero = tzero.item()

    self.outname_2_tensors[f"{self.name}_in.txt"] = tin #files
    self.outname_2_tensors[f"{self.name}_goldenout.txt"] = tout
  
  def get_kernel_line(self) -> str:
    graph = "QuantizeLinearGraph"
    kernel = "QuantizeLinearScalar"
    return f"{graph}<{kernel},{self.WINDOW_SIZE}> {self.name};"

  def get_arg_line(self) -> str:
    return f"float {self.name}_scale,\n{ctype} {self.name}_zero"
  
  def get_initlist_line(self) -> str:
    return f"{self.name}({self.name}_scale, {self.name}_zero)"
  
  def get_weight_line(self) -> str:
    ctype = dtype_to_cstr(self.dtype)
    return f"float {self.name}_scale {{{self.scale}}};\n" + \
      f"{ctype} {self.name}_zero {{{self.zero}}};"
  
  def get_callarg_line(self) -> str:
    return f"{self.name}_scale, {self.name}_zero"


class QLinearConvOp(OpParser):
  include_file: str = "graph_quantize_linear.h"

  def register_params(self, tensors: List[np.ndarray]):
    assert len(tensors) == 10
    tin, tin_scale, tin_zero, tw, tw_scale, tw_zero, tout_scale, tout_zero, tbias, tout = tensors

    inB, inC, inH, inW = tin.shape
    wM, wC, wK1, wK2 = tw.shape
    bM, = tbias.shape
    outB, outM, outH, outW = tout.shape
    
    assert inH == inW and outH == outW and inC == wC and wM == bM and bM == outM \
      and inB == outB and wK1 == wK2

    self.INP_W, self.OUT_W, self.B, self.C, self.M, self.K = inH, outH, inB, inC, wM, wK1 #config
    self.dtype = tout.dtype
    self.out_size = tout.size

    vector_size = VECTOR_WORD_BOUNDARY // tin.dtype.itemsize

    tw = pad_lastdim(tw, "QLinearConvOp weights", vector_size)
    if wK1 == 5:
      tw = tw[..., [5,5,5,5,0,0,1,1,2,2,3,3,4,4,5,5]]
    
    self.in_zero = tin_zero # heap
    self.weights, self.bias = tw, tbias

    # pad INP_W, OUT_W to vector boundary
    self.outname_2_tensors[f"{self.name}_in.txt"] = pad_lastdim(
      tin, "QLinearConvOp tin", vector_size, value=tin_zero) #files
    self.outname_2_tensors[f"{self.name}_goldenout.txt"] = pad_lastdim(
      tout, "QLinearConvOp tout", vector_size, value=tin_zero)
    
    import ipdb;ipdb.set_trace()
  
  def get_kernel_line(self) -> str:
    graph = "QLinearConvScalar"
    kernel = "QLinearConvGraph"
    return f"{graph}<{kernel},{self.INP_W},{self.OUT_W},{self.B},{self.C},{self.M},{self.K}> {self.name};"

  def get_arg_line(self) -> str:
    ctype = dtype_to_cstr(self.dtype)
    return f"{ctype} {self.name}_zero,\nstd::vector<{ctype}> {self.name}_w,\nstd::vector<{ctype}> {self.name}_b"
  
  def get_initlist_line(self) -> str:
    return f"{self.name}({self.name}_zero, {self.name}_w, {self.name}_b)"
  
  def get_weight_line(self) -> str:
    wstring = str(self.weights.flatten().tolist())[1:-1]
    bstring = str(self.bias.flatten().tolist())[1:-1]
    ctype = dtype_to_cstr(self.dtype)
    return f"{ctype} {self.name}_zero {self.in_zero};\n" + \
      f"std::vector<{ctype}> {self.name}_w {{{wstring}}};\n" + \
      f"std::vector<{ctype}> {self.name}_b {{{bstring}}};"
  
  def get_callarg_line(self) -> str:
    return f"{self.name}_zero, {self.name}_w, {self.name}_b"