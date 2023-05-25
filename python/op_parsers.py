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


def get_vector_boundary(tensor: np.ndarray):
  return VECTOR_WORD_BOUNDARY // tensor.dtype.itemsize


def get_tensor_type(tensor: np.ndarray):
  if tensor.shape == ():
    return dtype_to_cstr(tensor.dtype)
  else:
    return f"std::vector<{dtype_to_cstr(tensor.dtype)}>"


def save_tensor(output_path: str, 
                tensor: np.ndarray):
    n_lines = PLIO_WIDTH//tensor.dtype.itemsize
    if tensor.size % n_lines != 0:
      print(f"Warning: padding to write txt tensor.")
      tensor = tensor.flatten()
      tensor = pad_lastdim(tensor, "pad to write", n_lines)
    tensor = tensor.reshape(-1, n_lines)
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
    self.varname_2_tensors = {}  # generate files
    self.filename_2_tensors = {} # output txt
  
  def get_include_line(self) -> str:
    return f'#include "{self.include_file}"'
  
  def get_arg_line(self) -> str:
    return ",\n".join(
      f"{get_tensor_type(tensor)} {varname}" 
      for varname, tensor in self.varname_2_tensors.items())
  
  def get_initlist_line(self) -> str:
    initlist = ", ".join(varname for varname in self.varname_2_tensors)
    if initlist == "": 
      return ""
    return f"{self.name}({initlist})"
  
  def get_weight_line(self) -> str:
    lines = []
    for varname, tensor in self.varname_2_tensors.items():
      tensor_type = get_tensor_type(tensor)
      if "vector" in tensor_type:
        lines.append(f"{tensor_type} {varname} {{{str(tensor.flatten().tolist())[1:-1]}}};")
      else:
        lines.append(f"{tensor_type} {varname} = {str(tensor)};")
    return "\n".join(lines)
  
  def get_callarg_line(self) -> str:
    return ", ".join(varname for varname in self.varname_2_tensors)
  
  def save_txt(self, data_path: str):
    for outname, outtensor in self.filename_2_tensors.items():
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

    self.varname_2_tensors[f"{self.name}_w"] = pad_lastdim(tw, "Conv weights", 8) # heap
    self.varname_2_tensors[f"{self.name}_b"] = tbias

    self.filename_2_tensors[f"{self.name}_in.txt"] = tin # files
    self.filename_2_tensors[f"{self.name}_goldenout.txt"] = tout
  
  def get_kernel_line(self) -> str:
    graph = "ConvReluGraph"
    kernel = "ConvReluScalarBCHW"
    if self.OUT_W % 8 == 0 and self.K == 5:
      kernel = "Conv5x5on8ReluBCHW"
    
    return f"{graph}<{kernel},{self.INP_W},{self.OUT_W},{self.B},{self.C},{self.M},{self.K}> {self.name};"
  

class DequantizeLinearOp(OpParser):
  include_file: str = "graph_dequantize_linear.h"

  def register_params(self, tensors: List[np.ndarray]):
    assert len(tensors) == 4
    tin, tscale, tzero, tout = tensors

    assert tin.size == tout.size and tscale.shape == () and tzero.shape == ()
    
    self.WINDOW_SIZE = tin.size # config
    self.dtype = tout.dtype
    self.out_size = tout.size

    self.varname_2_tensors[f"{self.name}_scale"] = tscale # heap
    self.varname_2_tensors[f"{self.name}_zero"] = tzero
    
    self.filename_2_tensors[f"{self.name}_in.txt"] = tin #files
    self.filename_2_tensors[f"{self.name}_goldenout.txt"] = tout
  
  def get_kernel_line(self) -> str:
    graph = "DequantizeLinearGraph"
    kernel = "DequantizeLinearScalar"
    return f"{graph}<{kernel},{self.WINDOW_SIZE}> {self.name};"


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
    self.varname_2_tensors[f"{self.name}_w"] = pad_lastdim(tw, "Gemm weights", 4) # heap
    self.varname_2_tensors[f"{self.name}_b"] = tbias
  
    self.filename_2_tensors[f"{self.name}_in.txt"] = tin #files
    self.filename_2_tensors[f"{self.name}_goldenout.txt"] = tout

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

    self.filename_2_tensors[f"{self.name}_in.txt"] = tin #files
    self.filename_2_tensors[f"{self.name}_goldenout.txt"] = tout
  
  def get_kernel_line(self) -> str:
    graph = "MaxpoolGraph"
    kernel = "MaxpoolScalarBCHW"
    if self.OUT_W % 4 == 0 and self.dtype == "float32":
      kernel = "Maxpool2x2BCHW"
    return f"{graph}<{kernel},{dtype_to_cstr(self.dtype)},{self.INP_W},{self.OUT_W},{self.B},{self.C}> {self.name};"


class QGemm(OpParser):
  include_file: str = "graph_qgemm.h"
  _w: str = "_w"
  _b: str = "_b"
  _xzero: str = "_xzero"
  _wzero: str = "_wzero"
  _yzero: str = "_yzero"
  _xscale: str = "_xscale"
  _wscale: str = "_wscale"
  _yscale: str = "_yscale"

  def register_params(self, tensors: List[np.ndarray]):
    assert len(tensors) == 10
    tin, tin_scale, tin_zero, tw, tw_scale, tw_zero, tbias, tout_scale, tout_zero, tout = tensors

    inM, inK = tin.shape
    wN, wK = tw.shape
    bN, = tbias.shape
    outM, outN = tout.shape
    
    assert inM == outM and inK == wK and wN == bN and bN == outN

    self.M, self.K, self.N = inM, inK, wN # config
    self.dtype = tw.dtype
    self.out_size = tout.size
    self.tout = tout
  
    vector_size = get_vector_boundary(tin)
    
    self.varname_2_tensors[f"{self.name}_w"] = pad_lastdim(tw.T, "QGemm weights", vector_size) # heap
    self.varname_2_tensors[f"{self.name}_b"] = pad_lastdim(tbias, "QGemm bias", vector_size)
    self.varname_2_tensors[f"{self.name}_xscale"] = tin_scale
    self.varname_2_tensors[f"{self.name}_wscale"] = tw_scale
    self.varname_2_tensors[f"{self.name}_yscale"] = tout_scale
    self.varname_2_tensors[f"{self.name}_xzero"] = tin_zero
    self.varname_2_tensors[f"{self.name}_wzero"] = tw_zero
    self.varname_2_tensors[f"{self.name}_yzero"] = tout_zero
    
    self.NPAD = self.varname_2_tensors[f"{self.name}_w"].shape[-1]

    # pad INP_W, OUT_W to vector boundary
    self.filename_2_tensors[f"{self.name}_in.txt"] = pad_lastdim(
      tin, "QGemm tin", vector_size, value=tin_zero) #files
    self.filename_2_tensors[f"{self.name}_goldenout.txt"] = pad_lastdim(
      tout, "QGemm tout", vector_size, value=tin_zero)
    
  def get_kernel_line(self) -> str:
    kernel = "QgemmScalar"
    graph = "QgemmGraph"
    if self.NPAD%16==0 and self.K%4==0:
      kernel = "QgemmVector"
    return f"{graph}<{kernel},{self.M},{self.K},{self.N},{self.NPAD}> {self.name};"


class QLinearConvOp(OpParser):
  include_file: str = "graph_qlinearconv.h"

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

    vector_size = get_vector_boundary(tin)

    tw = pad_lastdim(tw, "QLinearConvOp weights", vector_size)
    if wK1 == 5:
      tw = tw[..., [5,5,5,5,0,0,1,1,2,2,3,3,4,4,5,5]]
    
    self.varname_2_tensors[f"{self.name}_w"] = tw # heap
    self.varname_2_tensors[f"{self.name}_b"] = tbias
    self.varname_2_tensors[f"{self.name}_xscale"] = tin_scale
    self.varname_2_tensors[f"{self.name}_wscale"] = tw_scale
    self.varname_2_tensors[f"{self.name}_yscale"] = tout_scale
    self.varname_2_tensors[f"{self.name}_xzero"] = tin_zero
    self.varname_2_tensors[f"{self.name}_wzero"] = tw_zero
    self.varname_2_tensors[f"{self.name}_yzero"] = tout_zero
    
    # pad INP_W, OUT_W to vector boundary
    tin = pad_lastdim(tin, "QLinearConvOp tin", vector_size, value=tin_zero) #files
    self.filename_2_tensors[f"{self.name}_in.txt"] = tin
    tout = pad_lastdim(tout, "QLinearConvOp tout", vector_size, value=tin_zero)
    self.filename_2_tensors[f"{self.name}_goldenout.txt"] = tout

    self.INP_W_PAD = tin.shape[-1]
    self.OUT_W_PAD = tout.shape[-1]
    
  def get_kernel_line(self) -> str:
    graph = "QLinearConvGraph"
    kernel = "QLinearConvVector"
    return f"{graph}<{kernel},{self.INP_W},{self.INP_W_PAD},{self.OUT_W},{self.OUT_W_PAD},{self.B},{self.C},{self.M},{self.K}> {self.name};"


class QuantizeLinearOp(OpParser):
  include_file: str = "graph_quantize_linear.h"

  def register_params(self, tensors: List[np.ndarray]):
    assert len(tensors) == 4
    tin, tscale, tzero, tout = tensors

    assert tin.size == tout.size and tscale.shape == () and tzero.shape == ()
    
    self.WINDOW_SIZE = tin.size # config
    self.dtype = tout.dtype
    self.out_size = tout.size

    self.varname_2_tensors[f"{self.name}_scale"] = tscale # heap
    self.varname_2_tensors[f"{self.name}_zero"] = tzero

    self.filename_2_tensors[f"{self.name}_in.txt"] = tin #files
    self.filename_2_tensors[f"{self.name}_goldenout.txt"] = tout
  
  def get_kernel_line(self) -> str:
    graph = "QuantizeLinearGraph"
    kernel = "QuantizeLinearScalar"
    return f"{graph}<{kernel},{self.WINDOW_SIZE}> {self.name};"
