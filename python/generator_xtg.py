from parser import Parser


class XtgGenerator:

  def __init__(self,
               parser: Parser):
    self.p = parser
  
  def get_xtg_masters(self, is_dout: bool) -> str:
    masters = []
    for i, input_name, tensor in self.p.get_input_id_name_tensor():
      plin_name = f"plin{i}_{self.p.graph_name}_{input_name}"
      out_filename = self.p.get_filename(input_name, is_dout)
      masters.append(
        f'("{plin_name}", f"{{args.input_dir}}/{out_filename}", 64, ' + \
        f'"{str(self.p.modelin_2_tensor[input_name].dtype)}")'
      )
    return "    " + ",\n".join(masters).replace("\n", "\n    ")
  
  def get_xtg_slaves(self, is_dout: bool) -> str:
    slaves = []
    for i, op in self.p.get_output_id_op(include_optional_output=is_dout):
      iter_cnt = 1 if is_dout else self.p.data_count
      plout_name = f"plout{i}_{self.p.graph_name}_{op.name}"
      out_filename = self.p.get_filename(op.get_output_filename(), is_dout)
      slaves.append(
        f'("{plout_name}", f"{{args.output_dir}}/{out_filename}", ' + \
        f'64, "{str(op.dtype)}", {op.out_size}*{iter_cnt})'
      )
    return "    " + ",\n".join(slaves).replace("\n", "\n    ")
  
  def generate_xtg_python_str(self, is_dout: bool):
    return f"""
import argparse
import logging

from xtg_aie import ExternalTraffic

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', required=True)
  parser.add_argument('--output_dir', required=True)
  args = parser.parse_args()
  
  logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")

  master_list = [
{self.get_xtg_masters(is_dout=is_dout)}
  ]

  slave_list = [
{self.get_xtg_slaves(is_dout=is_dout)}
  ]
  
  design = ExternalTraffic(master_list, slave_list)
  design.run()
"""

  def generate_xtg_python(self):
    with open(f"../design/trafficgen/xtg_{self.p.graph_name}.py", "w") as f:
      f.write(self.generate_xtg_python_str(is_dout=False))
    with open(f"../design/trafficgen/xtg_{self.p.graph_name}_output_inter.py", "w") as f:
      f.write(self.generate_xtg_python_str(is_dout=True))
