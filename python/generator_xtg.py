from typing import List
from parser import Parser


class XtgGenerator:

  def __init__(self,
               parser: Parser):
    self.parser = parser
    self.g = parser.graphs[0]
  
  def get_xtg_masters(self, is_e2e: bool) -> str:
    masters = []
    for op in self.g.in_ops:
      assert len(op.get_input_filenames()) == 1
      plin_name = f"plin{op.id}_{self.g.name}_{op.name}"
      input_filename = self.parser.parse_filename(op.get_input_filenames()[0], is_e2e=is_e2e)
      masters.append(
        f'("{plin_name}", f"{{args.input_dir}}/{input_filename}", 64, ' + \
        f'"{str(op.tensor.dtype)}")'
      )
    return "    " + ",\n".join(masters).replace("\n", "\n    ")
  
  def get_xtg_slaves(self, is_e2e: bool) -> str:
    slaves = []

    op_list = self.g.out_ops if is_e2e else self.g.out_ops + self.g.optout_ops
    
    for op in op_list:
      assert len(op.get_output_filenames()) == 1
      iter_cnt = self.g.data_count if is_e2e else 1
      
      plout_name = f"plout{op.id}_{self.g.name}_{op.name}"
      output_filename = self.parser.parse_filename(op.get_output_filenames()[0], is_e2e=is_e2e)
      slaves.append(
        f'("{plout_name}", f"{{args.output_dir}}/{output_filename}", ' + \
        f'64, "{str(op.dtype)}", {op.out_size}*{iter_cnt})'
      )
    return "    " + ",\n".join(slaves).replace("\n", "\n    ")
  
  def generate_xtg_python_str(self, is_e2e: bool):
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
{self.get_xtg_masters(is_e2e=is_e2e)}
  ]

  slave_list = [
{self.get_xtg_slaves(is_e2e=is_e2e)}
  ]
  
  design = ExternalTraffic(master_list, slave_list)
  design.run()
"""

  def generate_xtg_python(self):
    with open(f"../design/trafficgen/xtg_{self.g.name}.py", "w") as f:
      f.write(self.generate_xtg_python_str(is_e2e=True))
    with open(f"../design/trafficgen/xtg_{self.g.name}_output_inter.py", "w") as f:
      f.write(self.generate_xtg_python_str(is_e2e=False))
