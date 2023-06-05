from parser import Parser


class CfgGenerator:

  def __init__(self,
               parser: Parser):
    self.p = parser
  
  def get_cfg_input_kernels(self) -> str:
    mm2s_names = {}
    for i, tensor in enumerate(self.p.modelin_2_tensor.values()):
      dtype = tensor.dtype
      if dtype not in mm2s_names:
        mm2s_names[dtype] = []
      mm2s_names[dtype].append(f"{dtype}_mm2s_{i}")
    
    header = ""
    for dtype, typed_mm2s_names in mm2s_names.items():
      header += f"nk={dtype}_mm2s:{len(typed_mm2s_names)}:{','.join(typed_mm2s_names)}"
      header += "\n"
    
    return header
  
  def get_cfg_output_kernels(self, is_dout: bool) -> str:
    s2mm_names = {}
    ops = list(self.p.modelout_2_op.values())
    if is_dout:
      ops += [op for op in self.p.op_list if op not in self.p.modelout_2_op.values()]
    for i, op in enumerate(ops):
      dtype = str(op.dtype)
      if dtype not in s2mm_names: 
        s2mm_names[dtype] = []
      s2mm_names[dtype].append(f"{dtype}_s2mm_{i}")

    header = ""
    for dtype, typed_s2mm_names in s2mm_names.items():
      header += f"nk={dtype}_s2mm:{len(typed_s2mm_names)}:{','.join(typed_s2mm_names)}"
      header += "\n"
    
    return header
  
  def get_cfg_input_scs(self) -> str:
    in_scs = [
      f"stream_connect={tensor.dtype}_mm2s_{i}.s:ai_engine_0.plin{i}_{self.p.graph_name}_{inp_name}"
      for i, (inp_name, tensor) in enumerate(self.p.modelin_2_tensor.items())
    ]
    return "\n".join(in_scs)

  def get_cfg_output_scs(self, is_dout: bool) -> str:
    ops = list(self.p.modelout_2_op.values())
    if is_dout:
      ops += [op for op in self.p.op_list if op not in self.p.modelout_2_op.values()]
    out_scs = [
      f"stream_connect=ai_engine_0.plout{i}_{self.p.graph_name}_{op.name}:{op.dtype}_s2mm_{i}.s"
      for i, op in enumerate(ops)
    ]
    return "\n".join(out_scs)  
  
  def generate_cfg_str(self, is_dout: bool):
    return f"""
[connectivity]
{self.get_cfg_input_kernels()}
{self.get_cfg_output_kernels(is_dout=is_dout)}

#Connections For Insts 0...
{self.get_cfg_input_scs()}
{self.get_cfg_output_scs(is_dout=is_dout)}

[advanced]
# Disable Profiling in hw_emu so that it is faster...
param=hw_emu.enableProfiling=false
"""

  def generate_cfg(self):
    with open(f"../design/system_configs/{self.p.graph_name}.cfg", "w") as f:
      f.write(self.generate_cfg_str(is_dout=False))
    with open(f"../design/system_configs/{self.p.graph_name}_output_inter.cfg", "w") as f:
      f.write(self.generate_cfg_str(is_dout=True))
