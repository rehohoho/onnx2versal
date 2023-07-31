import numpy as np
from sklearn import metrics

from check import load_txt


# IC
ic_pred = load_txt("reports_dir/tiny_ic_int8/sw_emu/x86simulator_output/k018dequantizeLinear_goldenout_shape1000x10_host.txt")
ic_golden = load_txt("data/k018dequantizeLinear_goldenout_shape1000x10_host.txt")
ic_label = np.load("data/tiny_ic/Y_test.npy")
ic_top1 = (ic_pred.reshape(1000,-1)[:,:10].argmax(1) == ic_label).sum() / 1000 * 100
ic_match_count = (ic_pred.reshape(1000,-1)[:,:10].argmax(1) == ic_golden.reshape(1000,-1).argmax(1)).sum()
print(f"Top 1 (IC): {ic_top1}%\nGolden count (IC): {ic_match_count}/1000") # 86.8%, 992/1000

# KWS
kws_pred = load_txt("reports_dir/tiny_kws_int8/sw_emu/x86simulator_output/k015dequantizeLinear_goldenout_shape1000x12_host.txt")
kws_golden = load_txt("data/k015dequantizeLinear_goldenout_shape1000x12_host.txt")
kws_label = np.load("data/tiny_kws/Y_test.npy")
kws_top1 = (kws_pred.reshape(1000,-1).argmax(1) == kws_label).sum() / 1000 * 100
kws_match_count = (kws_pred.reshape(1000,-1).argmax(1) == kws_golden.reshape(1000,-1).argmax(1)).sum() 
print(f"Top 1 (KWS): {kws_top1}%\nGolden count (KWS): {kws_match_count}/1000") # 91.1%, 999/1000

# AD
ad_pred = load_txt("reports_dir/tiny_ad_int8/sw_emu/x86simulator_output/k029dequantizeLinear_goldenout_shape14000x640_host.txt").reshape(-1,196,640)
ad_data = load_txt("data/input_1_host.txt").reshape(-1,196,640)
ad_label = np.load("data/tiny_ad/Y_test.npy")
ad_pred_score = np.mean(np.mean(np.square(ad_data - ad_pred), axis=-1), axis=-1)
roc = metrics.roc_auc_score(ad_label, ad_pred_score)
print(f"ROC (AD): {roc}") # 0.880244

# VWW
vww_pred = load_txt("reports_dir/tiny_vww_int8/sw_emu/x86simulator_output/k033dequantizeLinear_goldenout_shape1000x2_host.txt")
vww_golden = load_txt("data/k033dequantizeLinear_goldenout_shape1000x2_host.txt")
vww_label = np.load("data/tiny_vww/Y_test.npy")
vww_top1 = (vww_pred.reshape(1000,-1)[:,:2].argmax(1) == vww_label.argmax(1)).sum() / 1000 * 100
vww_match_count = (vww_pred.reshape(1000,-1)[:,:2].argmax(1) == vww_golden.reshape(1000,-1).argmax(1)).sum() 
print(f"Top 1 (VWW): {vww_top1}%\nGolden count (VWW): {vww_match_count}/1000") # 82.5%, 974/1000
