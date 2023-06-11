from typing import List, Mapping, Any
import math

import numpy as np

TILE_SIZE = 32768
MAX_PARAM_SIZE = TILE_SIZE // 4 # ping-pong buffer
MAX_FLOAT_PARAMS = 16384//4
PLIO_WIDTH = 64 // 8 # in bytes
VECTOR_WORD_BOUNDARY = 16 # in bytes


def round_away(x):
  x = np.round(x, 4) # rounds 4.499996 and 4.496 to 4.5 first
  a = np.abs(x)
  b = np.floor(a) + np.floor(2*(a%1))
  return np.sign(x)*b


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
      tensor = tensor.flatten()
      tensor = pad_lastdim(tensor, "to write", n_lines)
    print(f"Saving tensor of shape {tensor.shape} into {output_path}")
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


def get_shape_str(tensor: np.ndarray):
  return f"shape{'x'.join(str(i) for i in tensor.shape)}"


def get_attribute_dict(attributes: List[Any]):
  attr_d = {}
  for attr in attributes:
    if attr.type == 1: attr_d[attr.name] = attr.f # FLOAT
    elif attr.type == 2: attr_d[attr.name] = attr.i # INT
    else: raise NotImplementedError(f"Unknown type {attr.f}")
  return attr_d


def factor_int(n: int, 
               multiplier: int, 
               upper_bound: int):
  factor_pairs = []
  factor = 1
  while factor <= n ** 0.5:
    if n % factor == 0:
      factor_pairs.append((n//factor, factor))
    factor += 1
  
  factor_pairs = factor_pairs + [(f2, f1) for f1, f2 in factor_pairs[::-1]]

  for f1, f2 in factor_pairs:
    if f1 * multiplier <= upper_bound:
      return f1, f2
  raise ValueError(f"Unable to find factors f1, f2 of {n} such that f1*{multiplier} <= {upper_bound}")


class OpParser:
  include_file: str

  def __init__(self, name: str):
    self.name = name
    self.argname_2_tensor = {}   # used as args in .cpp
    self.gmioname_2_tensor = {}  # used as gmio in .cpp
    self.filename_2_tensor = {}  # output txt

    self.tout = None             # reference copy to compare with
    self.gmio_repeats = 0
  
  def get_include_line(self) -> str:
    return f'#include "{self.include_file}"'
  
  def get_arg_line(self) -> str:
    return ",\n".join(
      f"{get_tensor_type(tensor)} {argname}" 
      for argname, tensor in self.argname_2_tensor.items())
  
  def get_initlist_line(self) -> str:
    initlist = ", ".join(argname for argname in self.argname_2_tensor)
    if initlist == "": 
      return ""
    return f"{self.name}({initlist})"
  
  def get_weight_line(self) -> str:
    lines = []
    varname_2_tensor = dict(self.argname_2_tensor)
    varname_2_tensor.update(self.gmioname_2_tensor)
    for varname, tensor in varname_2_tensor.items():
      tensor_type = get_tensor_type(tensor)
      if "vector" in tensor_type:
        lines.append(f"{tensor_type} {varname} {{{str(tensor.flatten().tolist())[1:-1]}}};")
      else:
        lines.append(f"{tensor_type} {varname} = {str(tensor)};")
    return "\n".join(lines)
  
  def get_callarg_line(self) -> str:
    return ", ".join(argname for argname in self.argname_2_tensor)
  
  def get_connect_line(self, last_port: str) -> str:
    return f"adf::connect<> ({last_port}, {self.name}.pin[0]);"

  def get_output_filename(self) -> str:
    if len(self.filename_2_tensor) == 0:
      raise ValueError(f"No output filename for {self.name}.")
    return list(self.filename_2_tensor.keys())[-1]
  
  def save_txt(self, data_path: str):
    for outname, outtensor in self.filename_2_tensor.items():
      save_tensor(f"{data_path}/{outname}", outtensor)
  
  def get_kernel_line(self) -> str:
    pass
  
  def disable_input_pad(self):
    """Used for first node and skipped nodes (node after skip)"""
    print(f"Disabled input padding for {self.name}, may result in choosing scalar op instead of vector.")
  
  def disable_output_pad(self):
    """For last node and skipped nodes (node before skip)"""
    print(f"Disabled output padding for {self.name}, may result in choosing scalar op instead of vector.")


class ArgmaxOp(OpParser):
  include_file: str = "graph_argmax.h"


class ConcatOp(OpParser):
  include_file: str = "graph_concat.h"


class ConvOp(OpParser):
  """Use input tensor for INP_H, pad width to vector boundary for INP_W
  Use output tensor for OUT_H, pad width to vector boundary for OUT_W
  Expects MxCxKxK weights as per PyTorch, pads to  MxCxKxK' where K'%8=0
  """
  include_file: str = "graph_conv.h"

  def __init__(self, name: str, is_relu: bool):
    super().__init__(name)
    self.is_relu = is_relu

  def register_params(self, tensors: List[np.ndarray]):
    assert len(tensors) == 4
    tin, tw, tbias, tout = tensors

    inB, inC, inH, inW = tin.shape
    wM, wC, wK1, wK2 = tw.shape
    bM, = tbias.shape
    outB, outM, outH, outW = tout.shape
    
    assert inH == inW and outH == outW and inC == wC and wM == bM and bM == outM \
      and inB == outB and wK1 == wK2

    self.tout = tout # reference copy to check against to compress graph
    
    self.argname_2_tensor[f"{self.name}_w"] = pad_lastdim(tw, "Conv weights", 8) # heap
    self.argname_2_tensor[f"{self.name}_b"] = tbias

    tin = pad_lastdim(tin, "ConvOp tin", get_vector_boundary(tin)) #files
    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin # files
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout

    tout = pad_lastdim(tout, "ConvOp tout", get_vector_boundary(tout))
    
    self.INP_H, self.INP_W, self.OUT_H, self.OUT_W = inH, tin.shape[-1], outH, tout.shape[-1] # config
    self.B, self.C, self.M, self.K = inB, inC, wM, wK1
    self.dtype = tout.dtype
    self.out_size = tout.size
  
  def get_kernel_line(self) -> str:
    graph = "ConvReluGraph"
    kernel = "ConvReluScalarBCHW"
    if self.OUT_W % 8 == 0 and self.K == 5:
      kernel = "Conv5x5on8ReluBCHW"
    
    return f"{graph}<{kernel},{self.INP_W},{self.OUT_W},{self.B},{self.C},{self.M},{self.K},{int(self.is_relu)}> {self.name};"
  

class DequantizeLinearOp(OpParser):
  """Use input tensor, pad width to vector boundary for INP_SIZE
  Use output tensor for OUT_SIZE. Assumes only input has to meet vector boundaries.
  """
  include_file: str = "graph_dequantize_linear.h"

  def register_params(self, tensors: List[np.ndarray]):
    assert len(tensors) == 4
    tin, tscale, tzero, tout = tensors

    assert tin.size == tout.size and tscale.shape == () and tzero.shape == ()

    self.tout = tout # reference copy to check against to compress graph
    
    self.argname_2_tensor[f"{self.name}_scale"] = tscale # heap
    self.argname_2_tensor[f"{self.name}_zero"] = tzero
    
    tin = pad_lastdim(tin, "DequantizeLinearOp tin", get_vector_boundary(tin)) #files
    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout

    tout = pad_lastdim(tout, "DequantizeLinearOp tin", get_vector_boundary(tout)) # config
    self.INP_SIZE, self.OUT_SIZE = tin.size, tout.size # config
    self.dtype = tout.dtype
    self.out_size = tout.size
  
  def get_kernel_line(self) -> str:
    graph = "DequantizeLinearGraph"
    kernel = "DequantizeLinearScalar"
    return f"{graph}<{kernel},{self.INP_SIZE},{self.OUT_SIZE}> {self.name};"
  
  def disable_output_pad(self):
    self.OUT_SIZE = self.tout.size
    self.out_size = self.tout.size
    super().disable_output_pad()


class GemmOp(OpParser):
  include_file: str = "graph_gemm.h"

  def __init__(self, name: str, is_relu: bool):
    super().__init__(name)
    self.is_relu = is_relu

  def register_params(self, tensors: List[np.ndarray], attributes: List[Any]):
    assert len(tensors) == 4
    tin, tw, tbias, tout = tensors
    
    attr_d = get_attribute_dict(attributes)
    if attr_d.get("transA"): tin = tin.transpose()
    if attr_d.get("transB"): tw = tw.transpose()
    if attr_d.get("alpha", 1) != 1: raise NotImplementedError("Gemm alpha not implemented yet.")
    if attr_d.get("beta", 1) != 1: raise NotImplementedError("Gemm beta not implemented yet.")

    self.M, self.K = tin.shape
    _, self.N = tout.shape
    self.dtype = tout.dtype
    self.out_size = tout.size
    assert self.M == tout.shape[0] and (self.K, self.N) == tw.shape and (self.N, ) == tbias.shape

    self.unpadded_N = self.N
    self.tout = tout # reference copy to check against to compress graph

    tw = pad_lastdim(tw, "Gemm tw", get_vector_boundary(tw))
    tbias = pad_lastdim(tbias, "Gemm tbias", get_vector_boundary(tbias))
    tin = pad_lastdim(tin, "Gemm tin", get_vector_boundary(tin))

    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout
    
    self.is_stream = tw.size >= MAX_FLOAT_PARAMS * 8
    
    self.N = tw.shape[1] # config
    if self.is_stream:
      self.M, self.repeat = factor_int(self.M, self.N * self.dtype.itemsize, MAX_PARAM_SIZE)

      graph = "GemmReluStreamGraph"
      kernel = "GemmReluScalarMKNKStream"
      if self.K % 4 == 0 and self.N % 8 == 0:
        kernel = "GemmReluMKKNStream"
        tw = pad_lastdim(tw, "Gemm tw", 8)
        tw = tw.reshape(self.K, -1, 8).transpose(1,0,2)
        self.gmioname_2_tensor[f"{self.name}_w"] = tw
        self.gmio_repeats = (self.M // 4 + self.M % 4) * self.repeat
      else:
        self.gmioname_2_tensor[f"{self.name}_w"] = tw.transpose() # MKNK
        self.gmio_repeats = self.M * self.repeat
      self.argname_2_tensor[f"{self.name}_b"] = tbias
      
      self.kernel_type = f"{graph}<{kernel},{self.M},{self.K},{self.N},{int(self.is_relu)}>"
    
    else:
      graph = "GemmReluMkknChunkGraph"
      kernel = "GemmReluScalarMKKN"
      concat_kernel = "ConcatScalar"
      
      chunkSize = int(MAX_FLOAT_PARAMS/self.K//4*4)
      if chunkSize >= self.N: chunkSize = self.N
      if self.K % 2 == 0 and chunkSize % 4 == 0: 
        kernel = "GemmReluMKKN"
      if chunkSize % 8 == 0 and self.N % 4 == 0: 
        concat_kernel = "ConcatFloat"
      
      self.argname_2_tensor[f"{self.name}_w"] = tw
      self.argname_2_tensor[f"{self.name}_b"] = tbias
      
      self.M, self.repeat = factor_int(self.M, max(self.K, self.N) * self.dtype.itemsize, MAX_PARAM_SIZE)
      self.kernel_type = f"{graph}<{kernel},{concat_kernel},{chunkSize},{self.M},{self.K},{self.N},{int(self.is_relu)}>"
  
  def get_kernel_line(self) -> str:
    return f"{self.kernel_type} {self.name};"
  
  def get_initlist_line(self) -> str:
    initlist = ", ".join(argname for argname in self.argname_2_tensor)
    if initlist == "": 
      return ""
    return f"{self.name}({initlist}, {self.repeat})"
  
  def get_connect_line(self, last_port: str) -> str:
    if self.is_stream:
      connect_list = [f"adf::connect<> ({last_port}, {self.name}.pin[0]);"]
      connect_list += [f"adf::connect<> (gmio_{gmioname}.out[0], {self.name}.pin[{i+1}]);"
        for i, gmioname in enumerate(self.gmioname_2_tensor)
      ]
      return "\n".join(connect_list)
    return f"for (int i = 0; i < {self.kernel_type}::CHUNK_COUNT; i++)\n" + \
      f"  adf::connect<> ({last_port}, {self.name}.pin[i]);"
  
  def disable_output_pad(self):
    self.N = self.unpadded_N
    if not self.is_stream:
      self.argname_2_tensor[f"{self.name}_w"] = self.argname_2_tensor[f"{self.name}_w"][...,:self.N]
      self.argname_2_tensor[f"{self.name}_b"] = self.argname_2_tensor[f"{self.name}_b"][...,:self.N]
    self.out_size = self.tout.size
    super().disable_output_pad()


class MacOp(OpParser):
  include_file: str = "graph_mac.h"

  def __init__(self, name: str, is_relu: bool):
    super().__init__(name)
    self.is_relu = is_relu

  def register_params(self, tensors: List[np.ndarray]):
    assert len(tensors) == 4
    tin, tw, tbias, tout = tensors

    assert tin.shape == tout.shape and tw.ndim == tbias.ndim == 1 and \
      tin.shape[-1] == tw.shape[0] == tbias.shape[0] == tout.shape[-1]

    self.tout = tout # reference copy to check against to compress graph

    tw = pad_lastdim(tw, "Mac tw", get_vector_boundary(tw)) # heap
    self.argname_2_tensor[f"{self.name}_w"] = tw
    tbias = pad_lastdim(tbias, "Mac tbias", get_vector_boundary(tbias))
    self.argname_2_tensor[f"{self.name}_b"] = tbias

    tin = pad_lastdim(tin, "MacOp tin", get_vector_boundary(tin)) #files
    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin # files
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout

    tout = pad_lastdim(tout, "MacOp tout", get_vector_boundary(tout))
    
    self.B, self.W = math.prod(tin.shape[:-1]), tin.shape[-1] # config
    self.dtype = tout.dtype
    self.out_size = tout.size

    self.B, self.repeat = factor_int(self.B, self.W * self.dtype.itemsize, MAX_PARAM_SIZE) # batch window size

  def get_kernel_line(self) -> str:
    graph = "MacGraph"
    kernel = "MacScalar"
    if self.W % 8 == 0:
      kernel = "MacFloat"
    return f"{graph}<{kernel},{dtype_to_cstr(self.dtype)},{self.B},{self.W},{int(self.is_relu)}> {self.name};"
  
  def get_initlist_line(self) -> str:
    initlist = ", ".join(argname for argname in self.argname_2_tensor)
    if initlist == "": 
      return ""
    return f"{self.name}({initlist}, {self.repeat})"


class PoolOp(OpParser):
  """Use input tensor for INP_H, pads width to vector boundary for INP_W,
  Use output tensor for OUT_W, pads width to vector boundary for OUT_W
  """
  include_file: str = "graph_pool.h"
  
  def register_params(self, tensors: List[np.ndarray]):
    assert len(tensors) == 2
    
    tin, tout = tensors
    inB, inC, inH, inW = tin.shape
    outB, outC, outH, outW = tout.shape
    self.unpadded_OUT_W = outW
    assert inH == inW and outH == outW and inC == outC and inB == outB

    self.tout = tout # reference copy to check against to compress graph
    
    tin = pad_lastdim(tin, "PoolOp tin", get_vector_boundary(tin)) #files
    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout # shape of non-padded
 
    tout = pad_lastdim(tout, "PoolOp tout", get_vector_boundary(tout))
    assert tin.shape[:-2] == tout.shape[:-2] and tin.shape[-2] % tout.shape[-2] == 0
    
    self.INP_H, self.INP_W, self.OUT_H, self.OUT_W = inH, tin.shape[-1], outH, tout.shape[-1]
    self.B, self.C = inB, inC
    self.dtype = tout.dtype
    self.out_size = tout.size
  
  def get_kernel_line(self) -> str:
    graph = "MaxpoolGraph"
    kernel = "MaxpoolScalarBCHW"
    if self.OUT_W % 4 == 0 and self.INP_W//self.OUT_W == 2 and self.dtype == "float32":
      kernel = "Maxpool2x2FloatBCHW"
    elif self.INP_W % 16 == 0 and self.OUT_W % 16 == 0 and self.INP_W//self.OUT_W == 2 and self.dtype == "int8":
      kernel = "Maxpool2x2Int8BCHW"
    return f"{graph}<{kernel},{dtype_to_cstr(self.dtype)},{self.INP_H},{self.INP_W},{self.OUT_H},{self.OUT_W},{self.B},{self.C}> {self.name};"
  
  def disable_output_pad(self):
    self.OUT_W = self.unpadded_OUT_W
    self.out_size = self.tout.size
    super().disable_output_pad()


class QGemm(OpParser):
  """Use input tensor for M, pad width to vector boundary for K
  Use output tensor, pad width to vector boundary for N
  """
  include_file: str = "graph_qgemm.h"

  def register_params(self, tensors: List[np.ndarray], attributes: List[Any]):
    assert len(tensors) == 10
    tin, tin_scale, tin_zero, tw, tw_scale, tw_zero, tbias, tout_scale, tout_zero, tout = tensors

    attr_d = get_attribute_dict(attributes)
    if attr_d.get("transA"): tin = tin.transpose()
    if attr_d.get("transB"): tw = tw.transpose()
    if attr_d.get("alpha", 1) != 1: raise NotImplementedError("QGemm alpha not implemented yet.")
    if attr_d.get("beta", 1) != 1: raise NotImplementedError("QGemm beta not implemented yet.")

    inM, inK = tin.shape
    wK, wN = tw.shape
    bN, = tbias.shape
    outM, outN = tout.shape
    
    assert inM == outM and inK == wK and wN == bN and bN == outN

    self.tin_K = inK
    self.tout = tout # reference copy to check against to compress graph
  
    vector_size = get_vector_boundary(tin)
    tw = pad_lastdim(tw, "QGemm weights", vector_size) # heap
    tbias = pad_lastdim(tbias, "QGemm bias", vector_size)
    self.argname_2_tensor[f"{self.name}_w"] = tw
    self.argname_2_tensor[f"{self.name}_b"] = tbias
    self.argname_2_tensor[f"{self.name}_xscale"] = tin_scale
    self.argname_2_tensor[f"{self.name}_wscale"] = tw_scale
    self.argname_2_tensor[f"{self.name}_yscale"] = tout_scale
    self.argname_2_tensor[f"{self.name}_xzero"] = tin_zero
    self.argname_2_tensor[f"{self.name}_wzero"] = tw_zero
    self.argname_2_tensor[f"{self.name}_yzero"] = tout_zero
    
    tin = pad_lastdim(tin, "QGemm tin", vector_size, value=tin_zero) #files
    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout # shape of non-padded

    tout = pad_lastdim(tout, "QGemm tout", vector_size, value=tin_zero) # config
    self.M, self.K, self.N = inM, tin.shape[-1], tout.shape[-1]
    self.dtype = tout.dtype
    self.out_size = tout.size
    
  def get_kernel_line(self) -> str:
    kernel = "QgemmScalar"
    graph = "QgemmGraph"
    if self.N%16==0:
      kernel = "QgemmVector"
    return f"{graph}<{kernel},{self.M},{self.K},{self.N}> {self.name};"
  
  def disable_input_pad(self):
    self.K = self.tin_K


class QLinearConvOp(OpParser):
  """Use input tensor for INP_H, pad width to vector boundary for INP_W,
  Use output tensor for OUT_H, pad width to vector boundary for OUT_W
  """
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

    self.tout = tout # reference copy to check against to compress graph
    
    vector_size = get_vector_boundary(tin)

    tw = pad_lastdim(tw, "QLinearConvOp weights", vector_size)
    if wK1 == 5:
      tw = tw[..., [5,5,5,5,0,0,1,1,2,2,3,3,4,4,5,5]]
    
    self.argname_2_tensor[f"{self.name}_w"] = tw # heap
    self.argname_2_tensor[f"{self.name}_b"] = tbias
    self.argname_2_tensor[f"{self.name}_xscale"] = tin_scale
    self.argname_2_tensor[f"{self.name}_wscale"] = tw_scale
    self.argname_2_tensor[f"{self.name}_yscale"] = tout_scale
    self.argname_2_tensor[f"{self.name}_xzero"] = tin_zero
    self.argname_2_tensor[f"{self.name}_wzero"] = tw_zero
    self.argname_2_tensor[f"{self.name}_yzero"] = tout_zero
    
    # pad INP_W, OUT_W to vector boundary
    tin = pad_lastdim(tin, "QLinearConvOp tin", vector_size, value=tin_zero) #files
    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout # shape of non-padded
    
    tout = pad_lastdim(tout, "QLinearConvOp tout", vector_size, value=tin_zero) #config
    self.INP_H, self.INP_W, self.OUT_H, self.OUT_W = inH, tin.shape[-1], outH, tout.shape[-1] 
    self.B, self.C, self.M, self.K = inB, inC, wM, wK1
    self.dtype = tout.dtype
    self.out_size = tout.size
    
  def get_kernel_line(self) -> str:
    graph = "QLinearConvGraph"
    kernel = "QLinearConvVector"
    return f"{graph}<{kernel},{self.INP_H},{self.INP_W},{self.OUT_H},{self.OUT_W},{self.B},{self.C},{self.M},{self.K}> {self.name};"


class QLinearMacOp(OpParser):
  include_file: str = "graph_qlinearmac.h"

  def __init__(self, name: str, is_relu: bool):
    super().__init__(name)
    self.is_relu = is_relu

  def register_params(self, tensors: List[np.ndarray]):
    assert len(tensors) == 17
    tin, tin_scale, tin_zero, tw, tw_scale, tw_zero, tz_scale, tz_zero, _, _, _, tbias, tbias_scale, tbias_zero, tout_scale, tout_zero, tout = tensors

    assert tin.shape == tout.shape and tw.ndim == tbias.ndim == 1 and \
      tin.shape[-1] == tw.shape[0] == tbias.shape[0] == tout.shape[-1] and \
      tin_scale.shape == tin_zero.shape == tw_scale.shape == tw_zero.shape == \
      tbias_scale.shape == tbias_zero.shape == tout_scale.shape == tout_zero.shape == ()

    self.B, self.W = math.prod(tin.shape[:-1]), tin.shape[-1] # config
    self.dtype = tout.dtype
    self.out_size = tout.size
    self.tout = tout # reference copy to check against to compress graph

    tw = pad_lastdim(tw, "QLinearMacOp tw", get_vector_boundary(tw)) # heap
    tbias = pad_lastdim(tbias, "QLinearMacOp tbias", get_vector_boundary(tbias))
    tin = pad_lastdim(tin, "QLinearMacOp tin", get_vector_boundary(tin))
    self.argname_2_tensor[f"{self.name}_w"] = tw
    self.argname_2_tensor[f"{self.name}_b"] = tbias
    self.argname_2_tensor[f"{self.name}_xscale"] = tin_scale
    self.argname_2_tensor[f"{self.name}_wscale"] = tw_scale
    self.argname_2_tensor[f"{self.name}_bscale"] = tbias_scale
    self.argname_2_tensor[f"{self.name}_zscale"] = tz_scale
    self.argname_2_tensor[f"{self.name}_yscale"] = tout_scale
    self.argname_2_tensor[f"{self.name}_xzero"] = tin_zero
    self.argname_2_tensor[f"{self.name}_wzero"] = tw_zero
    self.argname_2_tensor[f"{self.name}_bzero"] = tbias_zero
    self.argname_2_tensor[f"{self.name}_zzero"] = tz_zero
    self.argname_2_tensor[f"{self.name}_yzero"] = tout_zero

    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin # files
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout

    self.W = tin.shape[-1]
    self.B, self.repeat = factor_int(self.B, self.W * self.dtype.itemsize, MAX_PARAM_SIZE) # batch window size

  def get_kernel_line(self) -> str:
    graph = "QlinearMacGraph"
    kernel = "QlinearMacScalar"
    if self.W % 16 == 0:
      kernel = "QlinearMac"
    return f"{graph}<{kernel},{self.B},{self.W},{int(self.is_relu)}> {self.name};"
  
  def get_initlist_line(self) -> str:
    initlist = ", ".join(argname for argname in self.argname_2_tensor)
    if initlist == "": 
      return ""
    return f"{self.name}({initlist}, {self.repeat})"


class QLinearSoftmaxOp(OpParser):
  """Use input tensor for INP_H, pad width to vector boundary for INP_W,
  Use output tensor for OUT_H, pad width to vector boundary for OUT_W
  """
  include_file: str = "graph_qlinearsoftmax.h"

  def register_lastdim(self, tensors: List[np.ndarray]):
    tin, tin_scale, tin_zero, tout_scale, tout_zero, tout = tensors
    self.INP_W = tin.shape[-1]

    self.argname_2_tensor[f"{self.name}_xscale"] = tin_scale # heap
    self.argname_2_tensor[f"{self.name}_yscale"] = tout_scale
    self.argname_2_tensor[f"{self.name}_xzero"] = tin_zero
    self.argname_2_tensor[f"{self.name}_yzero"] = tout_zero
    
    tin = pad_lastdim(tin, "QlinearsoftmaxOp tin", get_vector_boundary(tin), value=tin_zero) # files
    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout
    
    self.INP_H, self.INP_W_PAD = math.prod(tin.shape[:-1]), tin.shape[-1] # config

  def register_params(self, tensors: List[np.ndarray], attributes: List[Any]):
    assert len(tensors) == 6
    tin, tin_scale, tin_zero, tout_scale, tout_zero, tout = tensors

    assert tin.size == tout.size and tin_scale.shape == () and tin_zero.shape == () and \
      tout_scale.shape == () and tout_zero.shape == ()

    self.tout = tout # reference copy to check against to compress graph

    attr_d = get_attribute_dict(attributes)
    axis = attr_d.get("axis", -1)
    if axis == 0:
      raise NotImplementedError("QLinearSoftmax not implemented for axis that is not last.")
    elif axis == -1 or axis == tin.ndim - 1:
      self.register_lastdim(tensors)
    else:
      raise NotImplementedError("QLinearSoftmax not implemented for axis that is not last.")    
    
    self.dtype = tout.dtype
    self.out_size = tout.size
    
  def get_kernel_line(self) -> str:
    graph = "QlinearsoftmaxGraph"
    kernel = "QlinearsoftmaxScalar"
    if self.INP_W_PAD % 16 == 0:
      kernel = "QlinearsoftmaxSingleaxis" # accuracy option
    return f"{graph}<{kernel},{self.INP_H},{self.INP_W},{self.INP_W_PAD}> {self.name};"


class QuantizeLinearOp(OpParser):
  """Use input tensor for INP_H, INP_W, pads INP_W to vector boundary for OUT_W
  Assumes only output has to meet vector boundaries.
  """
  include_file: str = "graph_quantize_linear.h"

  def register_params(self, tensors: List[np.ndarray]):
    assert len(tensors) == 4
    
    tin, tscale, tzero, tout = tensors
    assert tin.size == tout.size and tscale.shape == () and tzero.shape == () and tin.dtype == "float32" and tout.dtype == "int8"

    self.inW = tin.shape[-1]
    self.tout = tout # reference copy to check against to compress graph

    self.argname_2_tensor[f"{self.name}_scale"] = tscale # heap
    self.argname_2_tensor[f"{self.name}_zero"] = tzero

    tin = pad_lastdim(tout, "QuantizeLinearOp tin", get_vector_boundary(tin)) # files
    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout # shape of non-padded
    
    tout = pad_lastdim(tout, "QuantizeLinearOp tout", get_vector_boundary(tout))
    assert tout.shape[:-1] == tin.shape[:-1]
    
    inH = math.prod(tin.shape[:-1]) # config
    self.INP_H, self.INP_W, self.OUT_W = inH, tin.shape[-1], tout.shape[-1]
    self.dtype = tout.dtype
    self.out_size = tout.size
  
  def get_kernel_line(self) -> str:
    graph = "QuantizeLinearGraph"
    kernel = "QuantizeLinearScalar"
    if self.INP_W % 4 == 0 and self.OUT_W % 16 == 0:
      kernel = "QuantizeLinearVector"
    return f"{graph}<{kernel},{self.INP_H},{self.INP_W},{self.OUT_W}> {self.name};"
  
  def disable_input_pad(self):
    self.INP_W = self.inW
    super().disable_input_pad()


class SoftmaxOp(OpParser):
  """Use input tensor for INP_H, INP_W, pads INP_W to vector boundary for OUT_W
  Assumes only output has to meet vector boundaries.
  """
  include_file: str = "graph_softmax.h"

  def register_params(self, tensors: List[np.ndarray]):
    assert len(tensors) == 2
    
    tin, tout = tensors
    assert tin.size == tout.size

    self.tout = tout # reference copy to check against to compress graph
    self.INP_W = tin.shape[-1]

    tin = pad_lastdim(tin, "SoftmaxOp tin", 8) # files
    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout
    
    tout = pad_lastdim(tout, "SoftmaxOp tout", 8)

    self.INP_H, self.INP_W_PAD = math.prod(tin.shape[:-1]), tin.shape[-1] # config
    self.dtype = tout.dtype
    self.out_size = tout.size
  
  def get_kernel_line(self) -> str:
    graph = "SoftmaxGraph"
    kernel = "SoftmaxScalar"
    if self.INP_W_PAD % 8 == 0 and self.INP_H % 2 == 0:
      kernel = "SoftmaxMultiaxis"
    elif self.INP_W_PAD % 8 == 0:
      kernel = "SoftmaxSingleaxis"
    return f"{graph}<{kernel},{self.INP_H},{self.INP_W},{self.INP_W_PAD}> {self.name};"
