from parser import Parser
from op_parsers import InputOp


class CfgGenerator:

  def __init__(self,
               parser: Parser):
    self.p = parser
  
  def get_cfg_input_kernels(self) -> str:
    mm2s_names = {}
    for id, _, tensor in self.p.get_input_id_name_tensor():
      dtype = tensor.dtype
      if dtype not in mm2s_names:
        mm2s_names[dtype] = []
      mm2s_names[dtype].append(f"{dtype}_mm2s_{id}")
    
    header = ""
    for dtype, typed_mm2s_names in mm2s_names.items():
      header += f"nk={dtype}_mm2s:{len(typed_mm2s_names)}:{','.join(typed_mm2s_names)}"
      header += "\n"
    
    return header
  
  def get_cfg_output_kernels(self, is_dout: bool) -> str:
    s2mm_names = {}
    for id, op in self.p.get_output_id_op(include_optional_output=is_dout):
      dtype = str(op.dtype)
      if dtype not in s2mm_names: 
        s2mm_names[dtype] = []
      s2mm_names[dtype].append(f"{dtype}_s2mm_{id}")

    header = ""
    for dtype, typed_s2mm_names in s2mm_names.items():
      header += f"nk={dtype}_s2mm:{len(typed_s2mm_names)}:{','.join(typed_s2mm_names)}"
      header += "\n"
    
    return header
  
  def get_cfg_input_scs(self) -> str:
    in_scs = [
      f"stream_connect={tensor.dtype}_mm2s_{id}.s:ai_engine_0.plin{id}_{self.p.graph_name}_{input_name}"
      for id, input_name, tensor in self.p.get_input_id_name_tensor()
    ]
    return "\n".join(in_scs)

  def get_cfg_output_scs(self, is_dout: bool) -> str:
    out_scs = [
      f"stream_connect=ai_engine_0.plout{id}_{self.p.graph_name}_{op.name}:{op.dtype}_s2mm_{id}.s"
      for id, op in self.p.get_output_id_op(include_optional_output=is_dout)
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
