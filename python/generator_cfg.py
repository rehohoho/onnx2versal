from parser import Parser


class CfgGenerator:

  def __init__(self,
               parser: Parser):
    self.parser = parser
    self.g = parser.graphs[0]
  
  def get_cfg_input_kernels(self) -> str:
    mm2s_names = {}
    for op in self.g.in_ops:
      dtype = op.tensor.dtype
      if dtype not in mm2s_names:
        mm2s_names[dtype] = []
      mm2s_names[dtype].append(f"{dtype}_mm2s_{op.id}")
    
    header = ""
    for dtype, typed_mm2s_names in mm2s_names.items():
      header += f"nk={dtype}_mm2s:{len(typed_mm2s_names)}:{','.join(typed_mm2s_names)}"
      header += "\n"
    
    return header
  
  def get_cfg_output_kernels(self, is_e2e: bool) -> str:
    s2mm_names = {}
    
    op_list = self.g.out_ops if is_e2e else self.g.out_ops + self.g.optout_ops

    for op in op_list:
      dtype = str(op.dtype)
      if dtype not in s2mm_names: 
        s2mm_names[dtype] = []
      s2mm_names[dtype].append(f"{dtype}_s2mm_{op.id}")

    header = ""
    for dtype, typed_s2mm_names in s2mm_names.items():
      header += f"nk={dtype}_s2mm:{len(typed_s2mm_names)}:{','.join(typed_s2mm_names)}"
      header += "\n"
    
    return header
  
  def get_cfg_input_scs(self) -> str:
    in_scs = [
      f"stream_connect={op.tensor.dtype}_mm2s_{op.id}.s:ai_engine_0.plin{op.id}_{self.g.name}_{op.name}"
      for op in self.g.in_ops
    ]
    return "\n".join(in_scs)

  def get_cfg_output_scs(self, is_e2e: bool) -> str:
    op_list = self.g.out_ops if is_e2e else self.g.out_ops + self.g.optout_ops
    out_scs = [
      f"stream_connect=ai_engine_0.plout{op.id}_{self.g.name}_{op.name}:{op.dtype}_s2mm_{op.id}.s"
      for op in op_list
    ]
    return "\n".join(out_scs)  
  
  def generate_cfg_str(self, is_e2e: bool):
    return f"""
[connectivity]
{self.get_cfg_input_kernels()}
{self.get_cfg_output_kernels(is_e2e=is_e2e)}

#Connections For Insts 0...
{self.get_cfg_input_scs()}
{self.get_cfg_output_scs(is_e2e=is_e2e)}

[advanced]
# Disable Profiling in hw_emu so that it is faster...
param=hw_emu.enableProfiling=false
"""

  def generate_cfg(self):
    with open(f"../design/system_configs/{self.g.name}.cfg", "w") as f:
      f.write(self.generate_cfg_str(is_e2e=True))
    with open(f"../design/system_configs/{self.g.name}_output_inter.cfg", "w") as f:
      f.write(self.generate_cfg_str(is_e2e=False))
