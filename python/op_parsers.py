from typing import List, Mapping, Any
import math

import numpy as np

TILE_SIZE = 32768
MAX_HEAP_SIZE = 31712
MAX_PARAM_SIZE = 32768 // 2 # ping-pong buffer, fit input and output ping pongs
PLIO_WIDTH = 64 // 8 # in bytes
VECTOR_WORD_BOUNDARY = 16 # in bytes
MAX_CHUNKS = 8


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
  elif np_dtype == "uint8":
    return "uint8_t"
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
    elif attr.type == 3: attr_d[attr.name] = attr.s # STRING
    elif attr.type == 7: attr_d[attr.name] = attr.ints # INTS
    else: raise NotImplementedError(f"Unknown type {attr.f}")
  return attr_d


def factor_int(n: int, 
               multiplier: int, 
               upper_bound: int,
               offset: int = 0,
               force_split_chunksize: int = -1):
  factor_pairs = []
  factor = 1
  while factor <= n ** 0.5:
    if n % factor == 0:
      factor_pairs.append((n//factor, factor))
    factor += 1
  
  factor_pairs = factor_pairs + [(f2, f1) for f1, f2 in factor_pairs[::-1]]

  if force_split_chunksize != -1:
    for i, (f1, f2) in enumerate(factor_pairs):
      if f1 < force_split_chunksize: break
      if f2 > MAX_CHUNKS: break
    f1, f2 = factor_pairs[max(i-1, 0)]
    if f1 * multiplier + offset <= upper_bound:
      return f1, f2
  
  for f1, f2 in factor_pairs:
    if f1 * multiplier + offset <= upper_bound:
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
    self.disable_last_file_output = False # stream is broadcasted to different number of buffers not supported
  
  def get_include_line(self) -> str:
    return f'#include "{self.include_file}"'
  
  def get_arg_line(self) -> str:
    return ",\n".join(
      f"{get_tensor_type(tensor)} {argname}" 
      for argname, tensor in self.argname_2_tensor.items())
  
  def get_initlist_line(self) -> str:
    initlist = [argname for argname in self.argname_2_tensor]
    if hasattr(self, "repeat"):
      initlist.append(str(self.repeat))
    if len(initlist) == 0: 
      return ""
    init_str = ", ".join(init for init in initlist)
    return f"{self.name}({init_str})"
  
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
  
  def get_connect_line(self, last_port: str, i: int = 0) -> str:
    return f"adf::connect<> ({last_port}, {self.name}.pin[{i}]);"

  def get_gmio_connect_line(self, i: int = 0) -> str:
    return "\n".join(f"adf::connect<> (gmio_{gmio_name}.out[0], {self.name}.pin[{gmio_idx+i}]);"
        for gmio_idx, gmio_name in enumerate(self.gmioname_2_tensor))

  def get_input_filename(self) -> str:
    if len(self.filename_2_tensor) == 0:
      raise ValueError(f"No input filename for {self.name}.")
    return list(self.filename_2_tensor.keys())[0]
  
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
    """Used for first node and skipped nodes (node after skip), 
    only required if dimensions padded affect weights"""
    print(f"Disabled input padding for {self.name}, may result in choosing scalar op instead of vector.")
  
  def disable_output_pad(self):
    """For last node and skipped nodes (node before skip),
    only required if dimensions padded affect weights"""
    print(f"Disabled output padding for {self.name}, may result in choosing scalar op instead of vector.")

  def get_input_shape(self) -> List[int]:
    return []
  
  def get_computation_count(self):
    return 0
  
  def get_input_bandwidth_bits(self):
    return 0
  
  def get_output_bandwidth_bits(self):
    return 0


class AddOp(OpParser):
  include_file: str = "graph_add.h"

  def __init__(self, name: str, is_relu: bool):
    super().__init__(name)
    self.is_relu = is_relu
  
  def register_params(self, tensors: List[np.ndarray]):
    assert len(tensors) == 3

    ta, tb, tout = tensors
    assert ta.shape == tb.shape == tout.shape

    self.tout = tout # reference copy to check against to compress graph
    self.dtype = tout.dtype
    self.disable_last_file_output = True

    self.filename_2_tensor[f"{self.name}_inA_{get_shape_str(ta)}.txt"] = ta
    self.filename_2_tensor[f"{self.name}_inB_{get_shape_str(tb)}.txt"] = tb
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout
    
    self.W = ta.size
    self.out_size = tout.size # host buffer sizes
  
  def get_kernel_line(self) -> str:
    graph = "AddGraph"
    kernel = "AddFloat" if self.W % 4 == 0 and self.dtype == "float32" else "AddScalar"
    return f"{graph}<{kernel},{dtype_to_cstr(self.dtype)},{self.W},{int(self.is_relu)}> {self.name};"

  def get_connect_line(self, last_port: str, i: int = 0) -> str:
    return f"adf::connect<> {self.name}_s{i} ({last_port}, {self.name}.pin[{i}]);\n" + \
      f"adf::fifo_depth({self.name}_s{i}) = {self.W//8};\n"

  def get_computation_count(self):
    return self.W
  
  def get_input_shape(self):
    return [self.W]


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

  def get_conv_targs(self):
    return f"{self.INP_H},{self.INP_W},{self.INP_W_PAD},{self.OUT_W},{self.OUT_W_PAD},{self.STEP_H},{self.STEP_W}," + \
      f"{self.B},{self.C},{self.M},{self.KH},{self.KW},{self.GROUP},{int(self.is_relu)}," + \
      f"{self.H0},{self.H1},{self.W0},{self.W1}>"
    
  def register_attributes(self, attributes: List[Any]):
    attr_d = get_attribute_dict(attributes)
    for i in attr_d.get("dilations"):
      if i != 1: raise NotImplementedError("Dilated convolution not implemented.")
    self.GROUP = attr_d.get("group", 1)
    self.STEP_H, self.STEP_W = attr_d.get("strides", [1, 1])
    self.H0, self.W0, self.H1, self.W1 = attr_d.get("pads", [0, 0, 0, 0])
  
  def register_shapes(self, 
                      tin: np.ndarray, 
                      tw: np.ndarray, 
                      tbias: np.ndarray, 
                      tout: np.ndarray):
    self.B, self.C, self.INP_H, self.INP_W = tin.shape
    self.M, _, self.KH, self.KW = tw.shape
    self.OUT_H = (self.INP_H + self.H0 + self.H1 - self.KH) // self.STEP_H + 1
    self.OUT_W = (self.INP_W + self.W0 + self.W1 - self.KW) // self.STEP_W + 1
    
    vector_boundary = 8 if self.OUT_W > 4 else 4
    self.INP_W_PAD = (tin.shape[-1] + (vector_boundary-1)) // vector_boundary * vector_boundary
    self.OUT_W_PAD = (tout.shape[-1] + (vector_boundary-1)) // vector_boundary * vector_boundary
    self.out_size = tout.size // self.OUT_W * self.OUT_W_PAD # host buffer sizes

    assert tw.shape == (self.M, self.C//self.GROUP, self.KH, self.KW) and \
      tbias.shape == (self.M, ) and \
      tout.shape == (self.B, self.M, self.OUT_H, self.OUT_W)
    
  def pad_and_save_files(self,
                         tin: np.ndarray,
                         tout: np.ndarray):
    tin = pad_lastdim(tin, "ConvOp tin", self.INP_W_PAD)
    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout
  
  def pick_graph(self):
    PAD_H = self.INP_H + self.H0 + self.H1
    PAD_W = max(self.INP_W + self.W0 + self.W1, self.INP_W_PAD)
    PAD_W = (PAD_W + 3)//4*4
    self.W1 = PAD_W - self.INP_W - self.W0

    self.pad_kernel = "Pad2DStreamInt8" if self.INP_W_PAD % 16 == 0 else "Pad2DWindowScalar"

    # HCHUNK = OUT_H' * strides + overlap, overlap = K - strides
    multiplier = self.B * self.C * PAD_W * self.STEP_H * self.dtype.itemsize
    overlap = self.KH - self.STEP_H
    offset = self.B * self.C * overlap * PAD_W * self.dtype.itemsize
    HCHUNK, _ = factor_int(self.OUT_H, multiplier, MAX_HEAP_SIZE, offset, force_split_chunksize=2*overlap)
    self.HCHUNK = HCHUNK * self.STEP_H + overlap
    
    if self.HCHUNK >= PAD_H - (self.STEP_H-1):
      self.graph = "ConvReluStreamGraph"
      self.disable_last_file_output = True
      assert self.B * self.C * PAD_H * PAD_W * self.dtype.itemsize <= 32768
    else:
      self.graph = "ConvReluChunkHPktStreamGraph" if self.HCHUNK - overlap*2 >= 0 else "ConvReluChunkHStreamGraph"
      self.split_kernel = "SplitFilterFloatPktStream" if self.HCHUNK - overlap*2 >= 0 else "SplitFilterFloatStream"
    
      LCNT = (PAD_H - self.HCHUNK) // (self.HCHUNK - overlap) + 1
      if LCNT > 8:
        raise NotImplementedError(F"ConvReluChunkHStreamGraph of LCNT {LCNT} not implemented. Maximum LCNT = 8.")
      
      if (self.KH - self.STEP_H) // 2 > 0:
        assert (self.INP_H+self.H0+self.H1-self.HCHUNK) % (self.HCHUNK - (self.KH - self.STEP_H)) == 0
      else:
        assert self.OUT_H % ((self.HCHUNK - (self.KH - self.STEP_H)) // self.STEP_H) == 0
  
  def register_params(self, tensors: List[np.ndarray], attributes: List[Any]):
    assert len(tensors) == 4
    tin, tw, tbias, tout = tensors
    self.register_attributes(attributes)
    self.register_shapes(tin, tw, tbias, tout)

    self.tout = tout # reference copy to check against to compress graph
    self.dtype = tout.dtype

    self.pad_and_save_files(tin, tout)
    self.pick_graph()
    
    kernel = None
    if self.STEP_H in [1,2] and self.STEP_W in [1,2]:
      if self.KH == self.KW == 1 and self.OUT_W_PAD == 4 and self.STEP_H == self.STEP_W == 1:
        kernel = "Conv1x1Out4ReluStream"
        tw = pad_lastdim(tw.reshape(self.M, -1), "Conv weights", get_vector_boundary(tw))
      elif self.KH == self.KW == 1 and self.GROUP == 1 and self.INP_W_PAD % 4 == 0 and (self.OUT_W_PAD % 8 == 0 and self.STEP_W == 1 or self.OUT_W_PAD % 4 == 0 and self.STEP_W == 2):
        kernel = "Conv1x1ReluPktStream" if self.graph == "ConvReluChunkHPktStreamGraph" else "Conv1x1ReluStream"
        tw = pad_lastdim(tw.reshape(self.M, -1), "Conv weights", get_vector_boundary(tw))
      elif self.KW <= 4 and self.OUT_W_PAD == 4 and self.STEP_H == self.STEP_W == 1:
        kernel = "ConvHx4Out4ReluStream"
        tw = pad_lastdim(tw, "Conv weights", 4)
      elif self.KW <= 4 and self.INP_W_PAD % 4 == 0 and (self.OUT_W_PAD % 8 == 0 and self.STEP_W == 1 or self.OUT_W_PAD % 4 == 0 and self.STEP_W == 2):
        kernel = "ConvHx4ReluPktStream" if self.graph == "ConvReluChunkHPktStreamGraph" else "ConvHx4ReluStream"
        tw = pad_lastdim(tw, "Conv weights", 4)
      elif self.KW <= 6 and self.INP_W_PAD % 4 == 0 and self.OUT_W_PAD % 8 == 0 and self.STEP_H == 1 and self.STEP_W == 1:
        tw = pad_lastdim(tw, "Conv weights", get_vector_boundary(tw))
        kernel = "ConvHx8ReluStream"
    
    if kernel is None:
      kernel = "ConvReluScalarStream"
      if self.graph == "ConvReluChunkHPktStreamGraph":
        self.graph = "ConvReluChunkHStreamGraph"
        self.split_kernel = "SplitFilterFloatStream"
    
    self.gmioname_2_tensor[f"{self.name}_w"] = tw
    self.gmio_repeats = 1
    self.argname_2_tensor[f"{self.name}_b"] = tbias

    if self.graph == "ConvReluStreamGraph":
      self.kernel_type = f"{self.graph}<{kernel},{self.get_conv_targs()}"
    else:
      self.kernel_type = f"{self.graph}<{self.split_kernel},{kernel},ConcatFloatStream,{self.HCHUNK},{self.get_conv_targs()}"
  
  def get_kernel_line(self) -> str:
    return f"{self.kernel_type} {self.name};"
  
  def get_computation_count(self):
    return self.B * self.M * self.C * self.KH * self.KW * self.OUT_H * self.OUT_W_PAD
  
  def get_input_shape(self):
    return [self.B, self.C, self.INP_H, self.INP_W_PAD]
  

class DequantizeLinearOp(OpParser):
  """Use input tensor, pad width to vector boundary for INP_SIZE
  """
  include_file: str = "graph_dequantize_linear.h"

  def register_params(self, tensors: List[np.ndarray]):
    assert len(tensors) == 4
    tin, tscale, tzero, tout = tensors

    assert tin.size == tout.size and tscale.shape == () and tzero.shape == ()

    self.tout = tout # reference copy to check against to compress graph
    self.in_dtype = tin.dtype
    self.dtype = tout.dtype
    
    self.argname_2_tensor[f"{self.name}_scale"] = tscale # heap
    self.argname_2_tensor[f"{self.name}_zero"] = tzero
    
    tin = pad_lastdim(tin, "DequantizeLinearOp tin", get_vector_boundary(tin)) #files
    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout

    tout = pad_lastdim(tout, "DequantizeLinearOp tin", get_vector_boundary(tout)) # config
    self.B, self.INP_W = tin.shape[0], math.prod(tin.shape[1:])
    self.OUT_W = tout.shape[-1]

    self.B, self.repeat = factor_int(self.B, self.OUT_W*self.dtype.itemsize, MAX_PARAM_SIZE) # batch window size
    self.out_size = tout.size # host buffer sizes
  
  def get_kernel_line(self) -> str:
    if self.INP_W % 16 == 0 and self.OUT_W % 4 == 0:
      return f"DequantizeLinearGraph<DequantizeLinear,{dtype_to_cstr(self.in_dtype)},{self.B},{self.INP_W},{self.OUT_W}> {self.name};"
    return f"DequantizeLinearGraph<DequantizeLinearScalar,{dtype_to_cstr(self.in_dtype)},{self.B},{self.INP_W},{self.OUT_W}> {self.name};"

  def get_computation_count(self):
    return self.B * self.OUT_W
  
  def get_input_shape(self):
    return [self.B, self.INP_W]


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

    self.tout = tout # reference copy to check against to compress graph
    self.dtype = tout.dtype
    self.tensors = tin, tw, tbias, tout
    self.register()

  def register(self):
    tin, tw, tbias, tout = self.tensors
    self.M, self.K = tin.shape
    _, self.N = tout.shape
    assert self.M == tout.shape[0] and (self.K, self.N) == tw.shape and (self.N, ) == tbias.shape
    
    tin = pad_lastdim(tin, "Gemm tin", get_vector_boundary(tin))
    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout
    self.out_size = tout.size # host buffer sizes

    tw = pad_lastdim(tw, "Gemm tw", 8)
    tbias = pad_lastdim(tbias, "Gemm tbias", 8)
    self.N = tw.shape[-1]
    
    # window kernels
    if tw.nbytes <= MAX_PARAM_SIZE or tw.nbytes <= MAX_PARAM_SIZE * 8 and MAX_PARAM_SIZE//(tw.nbytes//self.N) //8*8 > 0:
      
      self.M, self.repeat = factor_int(self.M, max(self.K, self.N) * self.dtype.itemsize, MAX_PARAM_SIZE, 0, force_split_chunksize=32)
      self.argname_2_tensor[f"{self.name}_w"] = tw
      self.argname_2_tensor[f"{self.name}_b"] = tbias

      if tw.nbytes <= MAX_PARAM_SIZE:
        kernel = "GemmReluMKKN" if self.K % 4 == 0 and self.N % 8 == 0 else "GemmReluScalarMKKN"
        self.kernel_type = f"GemmReluGraph<{kernel},{self.M},{self.K},{self.N},{int(self.is_relu)}>"
      else:
        chunkSize = (self.N//MAX_CHUNKS +7) //8*8
        kernel = "GemmReluMKKN" if self.K % 4 == 0 and chunkSize % 8 == 0 else "GemmReluScalarMKKN"
        concat_kernel = "ConcatFloatStream" if self.dtype == "float32" else "ConcatInt8Stream"
        self.kernel_type = f"GemmReluMkknChunkGraph<{kernel},{concat_kernel},{chunkSize},{self.M},{self.K},{self.N},{int(self.is_relu)}>"
      
    # stream kernels
    else:
      
      self.M, repeat = factor_int(self.M, self.N * self.dtype.itemsize, MAX_PARAM_SIZE, 0, force_split_chunksize=32)
      self.argname_2_tensor[f"{self.name}_b"] = tbias
      
      nchunk, nchunk_count = factor_int(self.N, 0, 1, 0, force_split_chunksize=max(round(self.N / MAX_CHUNKS), 8))

      if (4*self.K + 3*nchunk) * 4 <= 24576:
        kernel = "GemmReluMKKNStream"
        self.gmio_repeats = (self.M // 4 + self.M % 4) * repeat
      else:
        kernel = "GemmReluMKKNTwoAccsStream"
        self.gmio_repeats = (self.M // 2 + self.M % 2) * repeat
      
      if nchunk == self.N:
        self.gmioname_2_tensor[f"{self.name}_w"] = tw.reshape(self.K, -1, 8).transpose(1,0,2)
        self.gmio_repeats = (self.M // 2 + self.M % 2) * repeat
        self.kernel_type = f"GemmReluStreamGraph<{kernel},{self.M},{self.K},{self.N},{int(self.is_relu)}>"
      else:
        tw = tw.reshape(self.K, nchunk_count, nchunk).transpose(1,0,2)
        for i in range(nchunk_count):
          self.gmioname_2_tensor[f"{self.name}_w{i}"] = tw[i].reshape(self.K, -1, 8).transpose(1,0,2)
        self.kernel_type = f"GemmReluMkknChunkNStreamGraph<{kernel},ConcatFloatStream,{nchunk},{self.M},{self.K},{self.N},{int(self.is_relu)}>"
      
  def get_kernel_line(self) -> str:
    return f"{self.kernel_type} {self.name};"
  
  def get_computation_count(self):
    return self.M * self.K * self.N
  
  def get_input_shape(self):
    return [self.M, self.K]


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
    self.out_size = tout.size # host buffer sizes

    self.B, self.repeat = factor_int(self.B, self.W * self.dtype.itemsize, MAX_PARAM_SIZE) # batch window size

  def get_kernel_line(self) -> str:
    graph = "MacGraph"
    kernel = "MacScalar"
    if self.W % 8 == 0:
      kernel = "MacFloat"
    return f"{graph}<{kernel},{dtype_to_cstr(self.dtype)},{self.B},{self.W},{int(self.is_relu)}> {self.name};"
  
  def get_computation_count(self):
    return self.B * self.W


class PoolOp(OpParser):
  """Use input tensor for INP_H, pads width to vector boundary for INP_W,
  """
  include_file: str = "graph_pool.h"

  def __init__(self, name: str, reduction_mode: str = "max"):
    super().__init__(name)
    self.reduction_mode = reduction_mode
  
  def register_params(self, tensors: List[np.ndarray], attributes: List[Any]):
    assert len(tensors) == 2
    
    attr_d = get_attribute_dict(attributes)
    if attr_d.get("strides") != attr_d.get("kernel_shape"): raise ValueError(f"Attribute error, strides {attr_d['strides']} not equal to kernel {attr_d['kernel_shape']}")
    kernel_shape = attr_d.get("kernel_shape")
    if len(kernel_shape) != 2: raise NotImplementedError(f"Pool for kernel shape {kernel_shape} not supported. Only Pool2D supported.")
    self.KH, self.KW = kernel_shape
    
    tin, tout = tensors
    self.B, self.C, self.INP_H, self.INP_W = tin.shape
    _, _, self.OUT_H, self.OUT_W = tout.shape
    assert tout.shape == (self.B, self.C, self.OUT_H, self.OUT_W)

    self.tout = tout # reference copy to check against to compress graph
    
    vector_boundary = 8 if self.OUT_W > 4 else 4
    tin = pad_lastdim(tin, "PoolOp tin", vector_boundary) #files
    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout # shape of non-padded
    tout = pad_lastdim(tout, "QLinearPoolOp tout", vector_boundary)
    
    self.INP_W, self.OUT_W_PAD = tin.shape[-1], tout.shape[-1]
    self.dtype = tout.dtype
    self.out_size = tout.size # host buffer sizes
  
  def get_kernel_line(self) -> str:
    if self.reduction_mode == "max":
      kernel = "MaxpoolScalarBCHW"
      if self.INP_W//self.OUT_W == 2 and self.INP_W % 8 == 0 and self.OUT_W_PAD % 4 == 0 and self.KH == self.KW == 2 and self.dtype == "float32":
        kernel = "Maxpool2x2FloatBCHW"
        self.OUT_W = self.OUT_W_PAD
      elif self.INP_W//self.OUT_W == 2 and self.INP_W % 16 == 0 and self.OUT_W_PAD % 8 == 0 and self.KH == self.KW == 2 and self.dtype == "int8":
        kernel = "Maxpool2x2Int8BCHW"
        self.OUT_W = self.OUT_W_PAD
    elif self.reduction_mode == "avg":
      kernel = "AvgpoolScalarBCHW"
    else:
      raise NotImplementedError(f"Pool for reduction mode {self.reduction_mode} not implemented.")
    
    if self.B*self.C*self.INP_H*self.INP_W*self.dtype.itemsize <= MAX_PARAM_SIZE:
      return f"PoolGraph<{kernel},{dtype_to_cstr(self.dtype)},{self.INP_H},{self.INP_W},{self.OUT_H},{self.OUT_W},{self.B},{self.C},{self.KH},{self.KW}> {self.name};"
    else:
      split_kernel = "SplitScalar" if self.dtype == "float32" else "SplitInt8"
      concat_kernel = "ConcatFloatStream" if self.dtype == "float32" else "ConcatInt8Stream"
      CCHUNK, _ = factor_int(self.C, self.B*self.INP_H*self.INP_W*self.dtype.itemsize, MAX_PARAM_SIZE)
      return f"PoolChunkCGraph<{split_kernel},{kernel},{concat_kernel},{CCHUNK}," + \
        f"{dtype_to_cstr(self.dtype)},{self.INP_H},{self.INP_W},{self.OUT_H},{self.OUT_W},{self.B},{self.C},{self.KH},{self.KW}> {self.name};"
    
    def get_computation_count(self):
      return self.B * self.C * self.INP_H * self.INP_W


class QGemmOp(OpParser):
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

    self.M, self.K = tin.shape
    _, self.N = tw.shape
    
    assert tw.shape == (self.K, self.N) and tbias.shape == (self.N, ) \
      and tout.shape == (self.M, self.N)

    self.tout = tout # reference copy to check against to compress graph
    self.tw_dtype = tw.dtype
    self.dtype = tout.dtype
  
    vector_size = get_vector_boundary(tin)
    tbias = (tbias - (tw.astype(int) - tw_zero).sum(0) * tin_zero).astype(np.int32)
    
    tin = pad_lastdim(tin, "QGemm tin", vector_size, value=tin_zero) #files
    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout # shape of non-padded

    tout = pad_lastdim(tout, "QGemm tout", vector_size, value=tin_zero) # config
    self.K, self.N = tin.shape[-1], tout.shape[-1]
    self.out_size = tout.size # host buffer sizes

    # must pad both dims due to second mac for -wzero
    tw = np.pad(tw, ((0, self.K-tw.shape[0]), (0, self.N-tw.shape[1])), "constant", constant_values=tw_zero)
    tbias = pad_lastdim(tbias, "QGemm bias", vector_size)
    self.argname_2_tensor[f"{self.name}_w"] = tw
    self.argname_2_tensor[f"{self.name}_b"] = tbias
    self.argname_2_tensor[f"{self.name}_xscale"] = tin_scale
    self.argname_2_tensor[f"{self.name}_wscale"] = tw_scale
    self.argname_2_tensor[f"{self.name}_yscale"] = tout_scale
    self.argname_2_tensor[f"{self.name}_xzero"] = tin_zero
    self.argname_2_tensor[f"{self.name}_wzero"] = tw_zero
    self.argname_2_tensor[f"{self.name}_yzero"] = tout_zero

    if tin.nbytes > MAX_PARAM_SIZE or tout.nbytes > MAX_PARAM_SIZE:
      raise NotImplementedError(f"No QGemm implementation for input size {tin.nbytes} or output size {tout.nbytes}")
      
    kernel = "QgemmStream" if self.N%16==0 else "QgemmScalarStream"
    if tw.nbytes > TILE_SIZE * 8:
      raise NotImplementedError(f"No QGemm implementation for weight size {tw.nbytes}")

    chunkSize, _ = factor_int(self.N, self.K * self.dtype.itemsize, TILE_SIZE, force_split_chunksize=max(round(self.N / MAX_CHUNKS), 16))
    if chunkSize == self.N:
      self.kernel_type = f"QgemmStreamGraph<{kernel},{dtype_to_cstr(self.dtype)},{dtype_to_cstr(self.tw_dtype)},{self.M},{self.K},{self.N}>"    
    else:
      concat = "ConcatInt8Stream"
      self.kernel_type = f"QgemmChunkNStreamGraph<{kernel},{concat},{chunkSize},{dtype_to_cstr(self.dtype)},{dtype_to_cstr(self.dtype)},{self.M},{self.K},{self.N}>"
    
  def get_kernel_line(self) -> str:
    return f"{self.kernel_type} {self.name};"
  
  def get_computation_count(self):
    return self.M*self.K*self.N + self.M*self.N*2 # add bias then quantize


class QLinearAddOp(OpParser):
  include_file: str = "graph_qlinearadd.h"

  def __init__(self, name: str, is_relu: bool):
    super().__init__(name)
    self.is_relu = is_relu
  
  def register_params(self, tensors: List[np.ndarray]):
    assert len(tensors) == 9

    ta, ta_scale, ta_zero, tb, tb_scale, tb_zero, tout_scale, tout_zero, tout = tensors
    assert ta.shape == tb.shape == tout.shape and \
      ta_scale.shape == ta_zero.shape == \
      tb_scale.shape == tb_zero.shape == \
      tout_scale.shape == tout_zero.shape == ()

    self.tout = tout # reference copy to check against to compress graph
    self.dtype = tout.dtype
    self.disable_last_file_output = True

    self.argname_2_tensor[f"{self.name}_inA_scale"] = ta_scale
    self.argname_2_tensor[f"{self.name}_inB_scale"] = tb_scale
    self.argname_2_tensor[f"{self.name}_out_scale"] = tout_scale
    self.argname_2_tensor[f"{self.name}_inA_zero"] = ta_zero
    self.argname_2_tensor[f"{self.name}_inB_zero"] = tb_zero
    self.argname_2_tensor[f"{self.name}_out_zero"] = tout_zero

    self.filename_2_tensor[f"{self.name}_inA_{get_shape_str(ta)}.txt"] = ta
    self.filename_2_tensor[f"{self.name}_inB_{get_shape_str(tb)}.txt"] = tb
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout
    
    self.W = ta.size
    self.out_size = tout.size # host buffer sizes
  
  def get_kernel_line(self) -> str:
    graph = "QLinearAddGraph"
    kernel = "QLinearAddInt8"
    return f"{graph}<{kernel},{dtype_to_cstr(self.dtype)},{self.W},{int(self.is_relu)}> {self.name};"

  def get_connect_line(self, last_port: str, i: int = 0) -> str:
    print(f"WARNING: Using reconvergent Op {__class__}, make sure to allocate sufficient fifo. Allocated {self.W/4}")
    if i == 0:
      return f"adf::connect<> {self.name}_s{i} ({last_port}, {self.name}.pin[{i}]);\n" + \
          f"adf::fifo_depth({self.name}_s{i}) = {self.W//4};"
    return super().get_connect_line(last_port, i)
  
  def get_computation_count(self):
    return self.W*2 # quantize


class QLinearConvOp(OpParser):
  """Use input tensor for INP_H, pad width to vector boundary for INP_W,
  Use output tensor for OUT_H, pad width to vector boundary for OUT_W
  """
  include_file: str = "graph_qlinearconv.h"

  def get_graph_targs(self):
    return f"{dtype_to_cstr(self.dtype)},{dtype_to_cstr(self.tw_dtype)},{self.INP_H},{self.INP_W},{self.INP_W_PAD},{self.OUT_W},{self.OUT_W_PAD},{self.STEP_H},{self.STEP_W}," + \
        f"{self.B},{self.C},{self.M},{self.KH},{self.KW},{self.GROUP}," + \
        f"{self.H0},{self.H1},{self.W0},{self.W1}>"

  def register_attributes(self, attributes: List[Any]):
    attr_d = get_attribute_dict(attributes)
    for i in attr_d.get("dilations"):
      if i != 1: raise NotImplementedError("Dilated convolution not implemented.")
    self.GROUP = attr_d.get("group", 1)
    self.STEP_H, self.STEP_W = attr_d.get("strides", [1, 1])
    self.H0, self.W0, self.H1, self.W1 = attr_d.get("pads", [0, 0, 0, 0])

  def register_shapes(self, 
                      tin: np.ndarray, 
                      tw: np.ndarray, 
                      tbias: np.ndarray, 
                      tout: np.ndarray):
    self.B, self.C, self.INP_H, self.INP_W = tin.shape
    self.M, _, self.KH, self.KW = tw.shape
    self.OUT_H = (self.INP_H + self.H0 + self.H1 - self.KH) // self.STEP_H + 1
    self.OUT_W = (self.INP_W + self.W0 + self.W1 - self.KW) // self.STEP_W + 1
    
    vector_boundary = get_vector_boundary(tin)
    self.INP_W_PAD = (tin.shape[-1] + (vector_boundary-1)) // vector_boundary * vector_boundary
    self.OUT_W_PAD = (tout.shape[-1] + (vector_boundary-1)) // vector_boundary * vector_boundary
    self.out_size = tout.size // self.OUT_W * self.OUT_W_PAD # host buffer sizes

    assert tw.shape == (self.M, self.C//self.GROUP, self.KH, self.KW) and \
      tbias.shape == (self.M, ) and \
      tout.shape == (self.B, self.M, self.OUT_H, self.OUT_W)
  
  def pad_and_save_files(self,
                         tin: np.ndarray,
                         tout: np.ndarray,
                         tin_zero: int):
    # pad INP_W, OUT_W to vector boundary
    tin = pad_lastdim(tin, "QLinearConvOp tin", get_vector_boundary(tin), value=tin_zero) #files
    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout # shape of non-padded

  def pick_graph(self):
    PAD_H = self.INP_H + self.H0 + self.H1
    PAD_W = max(self.INP_W + self.W0 + self.W1, self.INP_W_PAD)
    PAD_W = (PAD_W + 15)//16*16
    self.W1 = PAD_W - self.INP_W - self.W0

    self.pad_kernel = "Pad2DStreamInt8" if self.INP_W_PAD % 16 == 0 else "Pad2DWindowScalar"
    multiplier = self.B * self.C * PAD_W * self.STEP_H * self.dtype.itemsize
    overlap = self.KH - self.STEP_H
    offset = self.B * self.C * overlap * PAD_W * self.dtype.itemsize
    HCHUNK, _ = factor_int(self.OUT_H, multiplier, MAX_HEAP_SIZE, offset, force_split_chunksize=2*overlap)
    self.HCHUNK = HCHUNK * self.STEP_H + overlap

    if self.HCHUNK >= PAD_H - (self.STEP_H-1):
      self.graph = "QLinearConvStreamGraph"
      self.disable_last_file_output = True
      assert self.B * self.C * PAD_H * PAD_W * self.dtype.itemsize <= MAX_HEAP_SIZE
    else:
      self.graph = "QLinearConvChunkHPktStreamGraph" if self.HCHUNK - overlap*2 >= 0 else "QLinearConvChunkHStreamGraph"
      self.split_kernel = "SplitFilterInt8PktStream" if self.HCHUNK - overlap*2 >= 0 else "SplitInt8"
    
  def register_params(self, 
                      tensors: List[np.ndarray], 
                      attributes: List[Any]):
    assert len(tensors) == 10
    tin, tin_scale, tin_zero, tw, tw_scale, tw_zero, tout_scale, tout_zero, tbias, tout = tensors

    self.register_attributes(attributes)
    self.register_shapes(tin, tw, tbias, tout)

    self.tout = tout # reference copy to check against to compress graph
    self.tw_dtype = tw.dtype
    self.dtype = tout.dtype

    self.pad_and_save_files(tin, tout, tin_zero)
    self.pick_graph()
    
    tbias = tbias - tin_zero * (tw.astype(int) - tw_zero).reshape(self.M, -1).sum(1).astype(np.int32)

    kernel = "QLinearConvScalarStream"
    if self.KH == self.KW == 1 and self.GROUP == 1 and self.INP_W_PAD % 16 == 0 and self.OUT_W_PAD % 16 == 0 and self.STEP_H in [1,2] and self.STEP_W in [1,2]:
      tw = pad_lastdim(tw.reshape(self.M,-1), "QLinearConvOp weights", get_vector_boundary(tw))
      kernel = "QLinearConv1x1PktStream" if self.graph == "QLinearConvChunkHPktStreamGraph" else "QLinearConv1x1Stream"

    elif self.KW <= 4 and self.INP_W_PAD % 16 == 0 and self.OUT_W_PAD % 16 == 0 and self.STEP_H in [1,2,4] and self.STEP_W in [1,2,4]:
      tw = pad_lastdim(tw, "QLinearConvOp weights", 4)
      tw = pad_lastdim(tw.reshape(self.M, self.C//self.GROUP, -1), "QLinearConvOp weights", get_vector_boundary(tw))
      kernel = "QLinearConvHx4PktStream" if self.graph == "QLinearConvChunkHPktStreamGraph" else "QLinearConvHx4Stream" # QLinearConvHx4StreamScale32bit
    
    elif not (self.tw_dtype == self.dtype == "uint8") and self.KW <= 6 and self.INP_W_PAD % 16 == 0 and self.OUT_W_PAD % 16 == 0 and self.STEP_H == 1 and self.STEP_W == 1 and tw.size//self.KW*16 <= 65536 and self.graph != "QLinearConvChunkHPktStreamGraph":
      tw = pad_lastdim(tw, "QLinearConvOp weights", get_vector_boundary(tw))
      tw = tw[..., [15,15,15,15, 0,0,1,1, 2,2,3,3, 4,4,5,5]]
      kernel = "QLinearConvHx6x8bitStream"
    
    else:
      tw = pad_lastdim(tw.reshape(self.M, self.C//self.GROUP, -1), "QLinearConvOp weights", get_vector_boundary(tw))
      if self.graph == "QLinearConvChunkHPktStreamGraph":
        self.graph = "QLinearConvChunkHStreamGraph"
        self.split_kernel = "SplitInt8"

    self.gmioname_2_tensor[f"{self.name}_w"] = tw
    self.gmio_repeats = 1
  
    if self.graph == "QLinearConvStreamGraph":
      self.kernel_type = f"{self.graph}<{self.pad_kernel},{kernel},{self.get_graph_targs()}"
    elif self.graph in ["QLinearConvChunkHStreamGraph", "QLinearConvChunkHPktStreamGraph"]:
      self.kernel_type = f"{self.graph}<{self.split_kernel},{kernel},ConcatInt8Stream,{self.HCHUNK},{self.get_graph_targs()}"
    else:
      raise NotImplementedError(f"QLinearConv Graph {self.graph} not implemented.")
    
    self.argname_2_tensor[f"{self.name}_b"] = tbias
    self.argname_2_tensor[f"{self.name}_xscale"] = tin_scale
    self.argname_2_tensor[f"{self.name}_wscale"] = tw_scale
    self.argname_2_tensor[f"{self.name}_yscale"] = tout_scale
    self.argname_2_tensor[f"{self.name}_xzero"] = tin_zero
    self.argname_2_tensor[f"{self.name}_wzero"] = tw_zero
    self.argname_2_tensor[f"{self.name}_yzero"] = tout_zero
    
  def get_kernel_line(self) -> str:
    return f"{self.kernel_type} {self.name};"
  
  def get_computation_count(self):
    return self.B*self.C*self.OUT_H*self.OUT_W_PAD * (self.KH*self.KW + 1) # quantize


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
    self.tw_dtype = tw.dtype
    self.dtype = tout.dtype
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
    self.out_size = tout.size # host buffer sizes

  def get_kernel_line(self) -> str:
    graph = "QlinearMacGraph"
    kernel = "QlinearMacScalar"
    if self.W % 16 == 0:
      graph = "QlinearMacStreamGraph"
      kernel = "QlinearMac"
    else:
      self.B, self.repeat = factor_int(self.B, self.W * self.dtype.itemsize, MAX_PARAM_SIZE) # batch window size

    return f"{graph}<{kernel},{dtype_to_cstr(self.dtype)},{dtype_to_cstr(self.tw_dtype)},{self.B},{self.W},{int(self.is_relu)}> {self.name};"
  
  def get_computation_count(self):
    return self.B*self.W*3 # quantize twice


class QLinearPoolOp(OpParser):
  include_file: str = "graph_qlinearpool.h"

  def __init__(self, name: str, reduction_mode: str = "max"):
    super().__init__(name)
    self.reduction_mode = reduction_mode
  
  def register_params(self, tensors: List[np.ndarray], attributes: List[Any]):
    assert len(tensors) == 6
    
    attr_d = get_attribute_dict(attributes)
    if attr_d.get("strides") != attr_d.get("kernel_shape"): raise ValueError(f"Attribute error, strides {attr_d['strides']} not equal to kernel {attr_d['kernel_shape']}")
    kernel_shape = attr_d.get("kernel_shape")
    if len(kernel_shape) != 2: raise NotImplementedError(f"Pool for kernel shape {kernel_shape} not supported. Only Pool2D supported.")
    self.KH, self.KW = kernel_shape
    
    tin, tin_scale, tin_zero, tout_scale, tout_zero, tout = tensors

    self.B, self.C, self.INP_H, self.INP_W = tin.shape
    _, _, self.OUT_H, self.OUT_W = tout.shape
    self.unpadded_OUT_W = self.OUT_W
    assert tout.shape == (self.B, self.C, self.OUT_H, self.OUT_W) and \
      tin_scale.shape == tin_zero.shape == tout_scale.shape == tout_zero.shape == ()

    self.tout = tout # reference copy to check against to compress graph
    
    self.argname_2_tensor[f"{self.name}_inscale"] = tin_scale
    self.argname_2_tensor[f"{self.name}_outscale"] = tout_scale
    self.argname_2_tensor[f"{self.name}_inzero"] = tin_zero
    self.argname_2_tensor[f"{self.name}_outzero"] = tout_zero

    tin = pad_lastdim(tin, "QLinearPoolOp tin", get_vector_boundary(tin)) #files
    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout # shape of non-padded
 
    tout = pad_lastdim(tout, "QLinearPoolOp tout", get_vector_boundary(tout))
    assert tin.shape[:-2] == tout.shape[:-2] and tin.shape[-2] % tout.shape[-2] == 0
    
    self.INP_W, self.OUT_W = tin.shape[-1], tout.shape[-1]
    self.dtype = tout.dtype
    self.out_size = tout.size # host buffer sizes
  
  def get_kernel_line(self) -> str:
    if self.reduction_mode == "avg":
      kernel = "QLinearAvgpoolScalarBCHW"
    else:
      raise NotImplementedError(f"Pool for reduction mode {self.reduction_mode} not implemented.")
    
    if self.B*self.C*self.INP_H*self.INP_W*self.dtype.itemsize <= MAX_PARAM_SIZE:
      return f"QLinearPoolStreamGraph<{kernel},{dtype_to_cstr(self.dtype)},{self.INP_H},{self.INP_W},{self.OUT_H},{self.OUT_W},{self.B},{self.C},{self.KH},{self.KW}> {self.name};"
    else:
      split_kernel = "SplitScalar" if self.dtype == "float32" else "SplitInt8"
      concat_kernel = "ConcatFloatStream" if self.dtype == "float32" else "ConcatInt8Stream"
      CCHUNK, _ = factor_int(self.C, self.B*self.INP_H*self.INP_W*self.dtype.itemsize, MAX_PARAM_SIZE)
      return f"QLinearPoolChunkCStreamGraph<{split_kernel},{kernel},{concat_kernel},{CCHUNK}," + \
        f"{dtype_to_cstr(self.dtype)},{self.INP_H},{self.INP_W},{self.OUT_H},{self.OUT_W},{self.B},{self.C},{self.KH},{self.KW}> {self.name};"
    
  def disable_output_pad(self):
    self.OUT_W = self.unpadded_OUT_W
    self.out_size = self.tout.size
    super().disable_output_pad()
  
  def get_computation_count(self):
    return self.B*self.C*self.OUT_H*self.OUT_W * (self.KH*self.KW + 1) # quantize


class QLinearSoftmaxOp(OpParser):
  """Use original INP_H and INP_W, pad INP_W to vector boundary for INP_W_PAD,
  """
  include_file: str = "graph_qlinearsoftmax.h"

  def register_lastdim(self, tensors: List[np.ndarray]):
    tin, tin_scale, tin_zero, tout_scale, tout_zero, tout = tensors
    self.INP_W = tin.shape[-1]

    self.argname_2_tensor[f"{self.name}_xscale"] = tin_scale # heap
    self.argname_2_tensor[f"{self.name}_yscale"] = tout_scale
    self.argname_2_tensor[f"{self.name}_xzero"] = tin_zero
    self.argname_2_tensor[f"{self.name}_yzero"] = tout_zero
    
    tin = pad_lastdim(tin, "QLinearSoftmaxOp tin", get_vector_boundary(tin), value=tin_zero) # files
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
    self.out_size = tout.size # host buffer sizes
    
  def get_kernel_line(self) -> str:
    graph = "QLinearSoftmaxStreamGraph"
    kernel = "QLinearSoftmaxScalar"
    if self.INP_W_PAD % 16 == 0:
      kernel = "QLinearSoftmaxFloatmul" # accuracy option
    return f"{graph}<Pad2DStreamInt8,{kernel},{dtype_to_cstr(self.dtype)},{self.INP_H},{self.INP_W},{self.INP_W_PAD}> {self.name};"

  def get_computation_count(self):
    return self.INP_H*self.INP_W_PAD * 12 # mac, 8x mul_square, add, mac, srs


class QuantizeLinearOp(OpParser):
  """Use original INP_H, pads INP_W and OUT_W to vector boundary
  """
  include_file: str = "graph_quantize_linear.h"

  def register_params(self, tensors: List[np.ndarray]):
    assert len(tensors) == 4
    
    tin, tscale, tzero, tout = tensors
    assert tin.size == tout.size and tscale.shape == () and tzero.shape == () and tin.dtype == "float32" and tout.dtype in ["int8", "uint8"]

    self.inW = tin.shape[-1]
    self.tout = tout # reference copy to check against to compress graph

    self.argname_2_tensor[f"{self.name}_scale"] = tscale # heap
    self.argname_2_tensor[f"{self.name}_zero"] = tzero

    tin = pad_lastdim(tin, "QuantizeLinearOp tin", get_vector_boundary(tin)) # files
    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout # shape of non-padded
    
    tout = pad_lastdim(tout, "QuantizeLinearOp tout", get_vector_boundary(tout))
    assert tout.shape[:-1] == tin.shape[:-1]
    
    inH = math.prod(tin.shape[:-1]) # config
    self.INP_H, self.INP_W, self.OUT_W = inH, tin.shape[-1], tout.shape[-1]
    self.dtype = tout.dtype
    self.out_size = tout.size # host buffer sizes
  
  def get_kernel_line(self) -> str:
    if not (self.INP_W % 4 == 0 and self.OUT_W % 16 == 0):
      raise NotImplementedError(f"Expect INP_W%4==0, OUT_W%16==0, but got INP_W {self.INP_W}, OUT_W {self.OUT_W}")

    self.HCHUNK, _ = factor_int(self.INP_H, self.INP_W*4, TILE_SIZE, 0, force_split_chunksize=1)
    if self.HCHUNK == self.INP_H:
      return f"QuantizeLinearStreamGraph<QuantizeLinearFmulStream,{dtype_to_cstr(self.dtype)},{self.INP_H},{self.INP_W},{self.OUT_W}> {self.name};"
    return f"QuantizeLinearChunkHPktStreamGraph<QuantizeLinearFmulStream,{self.HCHUNK},{dtype_to_cstr(self.dtype)},{self.INP_H},{self.INP_W},{self.OUT_W}> {self.name};"
    
  
  def disable_input_pad(self):
    # self.INP_W = self.inW
    super().disable_input_pad()
  
  def get_computation_count(self):
    return self.INP_H*self.INP_W_PAD * 2
  
  def get_input_shape(self):
    return [self.INP_H, self.INP_W]


class SoftmaxOp(OpParser):
  """Use original INP_H and INP_W, pads INP_W to vector boundary for OUT_W
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
    
    tout = pad_lastdim(tout, "SoftmaxOp tout", get_vector_boundary(tout))

    self.INP_H, self.INP_W_PAD = math.prod(tin.shape[:-1]), tin.shape[-1] # config
    self.dtype = tout.dtype
    self.out_size = tout.size # host buffer sizes

    self.kernel = "SoftmaxScalar"
    if self.INP_W_PAD % 8 == 0 and self.INP_H % 2 == 0:
      self.kernel = "SoftmaxMultiaxis"
    elif self.INP_W_PAD % 8 == 0:
      self.kernel = "SoftmaxSingleaxis"
  
  def get_kernel_line(self) -> str:
    graph = "SoftmaxGraph"
    return f"{graph}<{self.kernel},{self.INP_H},{self.INP_W},{self.INP_W_PAD}> {self.name};"

  def get_computation_count(self):
    return self.INP_H*self.INP_W_PAD * 12 # mac, 8x mul_square, add, mac, srs
  
  def get_input_shape(self):
    return [self.INP_H, self.INP_W_PAD]


class TransposeOp(OpParser):
  include_file: str = "graph_transpose.h"

  def register_params(self, tensors: List[np.ndarray], attributes: List[Any]):
    assert len(tensors) == 2

    attr_d = get_attribute_dict(attributes)
    self.perm = attr_d.get("perm")
    
    tin, tout = tensors
    if self.perm == [0,3,1,2]:
      self.B, self.H, self.W, self.C = tin.shape
      self.PAD_W = self.W
      assert tout.shape == (self.B, self.C, self.H, self.W)
    elif self.perm == [0,2,3,1]:
      self.B, self.C, self.H, self.W = tin.shape
      assert tout.shape == (self.B, self.H, self.W, self.C)
      
      tin = pad_lastdim(tin, "SoftmaxOp tin", get_vector_boundary(tin)) # files
      self.PAD_W = tin.shape[-1]
    else:
      raise NotImplementedError(f"Transpose for {attr_d.get('perm')} not implemented yet.")

    self.tout = tout # reference copy to check against to compress graph

    self.filename_2_tensor[f"{self.name}_in_{get_shape_str(tin)}.txt"] = tin
    self.filename_2_tensor[f"{self.name}_goldenout_{get_shape_str(tout)}.txt"] = tout
    
    self.dtype = tout.dtype
    self.out_size = tout.size # host buffer sizes
  
  def get_kernel_line(self) -> str:
    kernel = "TransposeScalarBHWC2BCHW" if self.perm == [0,3,1,2] else "TransposeScalarBCHW2BHWC"
    if self.B*self.H*self.W*self.C*self.dtype.itemsize > TILE_SIZE and self.perm == [0,3,1,2] and self.dtype == "float32":
      HCHUNK, _ = factor_int(self.H, self.B*self.W*self.C*self.dtype.itemsize, TILE_SIZE)
      kernel = "TransposeScalarBHWC2BCHWStream"
      concat_kernel = "ConcatFloatStream" if self.dtype == "float32" else "ConcatInt8Stream"
      return f"TransposeHChunkGraph<{kernel},{concat_kernel},{HCHUNK},{dtype_to_cstr(self.dtype)},{self.B},{self.H},{self.W},{self.C},{self.PAD_W}> {self.name};"
    else:
      return f"TransposeGraph<{kernel},{dtype_to_cstr(self.dtype)},{self.B},{self.H},{self.W},{self.C},{self.PAD_W}> {self.name};"
  
  def get_computation_count(self):
    return self.B*self.H*self.W*self.C
  
  def get_input_shape(self):
    return [self.B, self.H, self.W, self.C]
