import numpy as np
from sklearn import metrics

from check import load_txt


# IC
def get_ic_result(pred, label, golden):
  top1 = (pred.reshape(1000,-1)[:,:10].argmax(1) == label).sum() / 1000 * 100
  match_count = (pred.reshape(1000,-1)[:,:10].argmax(1) == golden.reshape(1000,-1).argmax(1)).sum()
  print(f"Top 1 (IC): {top1}%\nGolden count (IC): {match_count}/1000")

ic_label = np.load("data/tiny_ic/Y_test.npy")
ic_pred = load_txt("reports_dir/tiny_ic_int8/sw_emu/x86simulator_output/k018dequantizeLinear_goldenout_shape1000x10_host.txt")
ic_golden = load_txt("data/k018dequantizeLinear_goldenout_shape1000x10_host.txt")
get_ic_result(ic_pred, ic_label, ic_golden) # 86.8%, 992/1000

# Small IC
ic_small_label = np.load("data/tiny_ic_small/Y_test.npy")[:1000]
ic_small_pred = load_txt("reports_dir/tiny_ic_small_int8/sw_emu/x86simulator_output/k011dequantizeLinear_goldenout_shape1000x10_host.txt")
ic_small_golden = load_txt("data/k011dequantizeLinear_goldenout_shape1000x10_host.txt")
get_ic_result(ic_small_pred, ic_small_label, ic_small_golden) # 86.8%, 992/1000


# KWS
def get_kws_result(pred, label, golden):
  top1 = (pred.reshape(1000,-1).argmax(1) == label).sum() / 1000 * 100
  match_count = (pred.reshape(1000,-1).argmax(1) == golden.reshape(1000,-1).argmax(1)).sum()
  print(f"Top 1 (KWS): {top1}%\nGolden count (KWS): {match_count}/1000")

kws_label = np.load("data/tiny_kws/Y_test.npy")
kws_pred = load_txt("reports_dir/tiny_kws_int8/sw_emu/x86simulator_output/k015dequantizeLinear_goldenout_shape1000x12_host.txt")
kws_golden = load_txt("data/k015dequantizeLinear_goldenout_shape1000x12_host.txt")
get_kws_result(kws_pred, kws_label, kws_golden) # 91.0%, 1000/1000

# Small KWS
kws_small_label = np.load("data/tiny_kws_small/Y_test.npy")
kws_small_pred = load_txt("reports_dir/tiny_kws_small_fused_int8/sw_emu/x86simulator_output/k006dequantizeLinear_goldenout_shape1000x12_host.txt")
kws_small_golden = load_txt("data/k006dequantizeLinear_goldenout_shape1000x12_host.txt")
get_kws_result(kws_small_pred, kws_small_label, kws_small_golden) # 84.8


# AD
def get_ad_result(pred, data, label):
  pred_score = np.mean(np.mean(np.square(data - pred), axis=-1), axis=-1)
  roc = metrics.roc_auc_score(label, pred_score)
  print(f"ROC (AD): {roc}") # 0.880244

ad_label = np.load("data/tiny_ad/Y_test.npy")
ad_pred = load_txt("reports_dir/tiny_ad_int8/sw_emu/x86simulator_output/k029dequantizeLinear_goldenout_shape14000x640_host.txt").reshape(-1,196,640)
ad_data = load_txt("data/input_1_host.txt").reshape(-1,196,640)
get_ad_result(ad_pred, ad_data, ad_label) # 0.880244

# Small AD
ad_small_label = np.load("data/tiny_ad_small/Y_test.npy")
ad_small_pred = load_txt("reports_dir/tiny_ad_small_fused_int8/sw_emu/x86simulator_output/k007dequantizeLinear_goldenout_shape196000x64_host.txt").reshape(-1,196,64)
ad_small_data = load_txt("data/input_1_host.txt").reshape(-1,196,64)
get_ad_result(ad_small_pred, ad_small_data, ad_small_label) # 0.830616


# VWW
def get_vww_result(pred, golden, label):
  top1 = (pred.reshape(1000,-1)[:,:2].argmax(1) == label.argmax(1)).sum() / 1000 * 100
  match_count = (pred.reshape(1000,-1)[:,:2].argmax(1) == golden.reshape(1000,-1).argmax(1)).sum() 
  print(f"Top 1 (VWW): {top1}%\nGolden count (VWW): {match_count}/1000")

vww_pred = load_txt("reports_dir/tiny_vww_int8/sw_emu/x86simulator_output/k033dequantizeLinear_goldenout_shape1000x2_host.txt")
vww_golden = load_txt("data/k033dequantizeLinear_goldenout_shape1000x2_host.txt")
vww_label = np.load("data/tiny_vww/Y_test.npy")
get_vww_result(vww_pred, vww_golden, vww_label) # 82.5%, 974/1000
