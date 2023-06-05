from parser import Parser


class XtgGenerator:

  def __init__(self,
               parser: Parser):
    self.p = parser
  
  def get_xtg_masters(self, is_dout: bool) -> str:
    masters = []
    for i, inp_name in enumerate(self.p.modelin_2_tensor):
      plin_name = f"plin{i}_{self.p.graph_name}_{inp_name}"
      out_filename = self.p.get_filename(inp_name, is_dout)
      masters.append(
        f'("{plin_name}", f"{{args.input_dir}}/{out_filename}", 64, ' + \
        f'"{str(self.p.modelin_2_tensor[inp_name].dtype)}")'
      )
    return "    " + ",\n".join(masters).replace("\n", "\n    ")
  
  def get_xtg_slaves(self, is_dout: bool) -> str:
    slaves = []
    for _, op in self.p.modelout_2_op.items():
      size = op.out_size if is_dout else op.out_size * self.p.data_count
      for i, op in enumerate(self.p.modelout_2_op.values()):
        plout_name = f"plout{i}_{self.p.graph_name}_{op.name}"
        out_filename = self.p.get_filename(op.get_output_filename(), is_dout)
        slaves.append(
          f'("{plout_name}", f"{{args.output_dir}}/{out_filename}", ' + \
          f'64, "{str(op.dtype)}", {size})'
        )

    if is_dout:
      i = len(self.p.modelout_2_op)
      for op in self.p.op_list:
        if op in self.p.modelout_2_op.values(): continue
        plout_name = f"plout{i}_{self.p.graph_name}_{op.name}"
        slaves.append(
          f'("{plout_name}", f"{{args.output_dir}}/{op.get_output_filename()}", ' + \
          f'64, "{str(op.dtype)}", {op.out_size})')
        i += 1
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
