# -*- coding: utf-8 -*-
"""坏 bin 取证：渲染谱的 rank-2 投影检验（bin 8/11/16 vs 好 bin）。"""
import contextlib
import io
import math
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_e2e_demo as demo                                 # noqa: E402
import plot_demo as pd                                      # noqa: E402
from synthts import calib, render_fft                       # noqa: E402
from synthts.cli import main as synth_main, nature_magnetic_amplitude  # noqa: E402
from synthts.model import load_site                         # noqa: E402

out = 'diag_tmp'
if os.path.isdir(out):
    shutil.rmtree(out)
os.makedirs(out)
demo.build_model(os.path.join(out, 'mtl.XMLSITE'), 'MTL', 100.0, 0.3, 0.0)
with contextlib.redirect_stdout(io.StringIO()):
    rc = synth_main(['-model', os.path.join(out, 'mtl.XMLSITE'), '-out', out,
                     '-dur', '3600', '-seed', '42', '-scale', '10',
                     '-dense', '2', '-source', 'noise'])
assert rc == 0

site = load_site(os.path.join(out, 'mtl.XMLSITE'))
sig = site.get_frequencies(False)
se = [nature_magnetic_amplitude(f) / math.sqrt(
    abs(site.fields[i].hx1) ** 2 + abs(site.fields[i].hy1) ** 2)
    for i, f in enumerate(sig)]
sb = [nature_magnetic_amplitude(f) / math.sqrt(
    abs(site.fields[i].hx2) ** 2 + abs(site.fields[i].hy2) ** 2)
    for i, f in enumerate(sig)]
site.update_nature_magnetic_amplitude(se, sb)

n = 1024
freqs = np.fft.rfftfreq(n, 1.0 / 15.0)
T = render_fft.transfer_matrix(site, freqs, scale=10.0)

# 读回该窗的数据（窗 0 = 前 1024 样本）
from synthts import mtprocess
_, data = None, None
sr, dat = mtprocess.read_ts(os.path.join(out, 'MTL.TS5'))
X = np.stack([dat[c][:n] for c in range(4)]).astype(np.float64)
V = [np.fft.rfft(X[c]) for c in range(4)]

# 跨窗单 bin 解 → median/mean 对照（坏 bin 诊断）
nwin = min(len(dat[0]) // n, 105)
print('nwin=%d' % nwin)
Zw = {k: [] for k in (8, 11, 16, 9, 15, 20)}
for i in range(nwin):
    Xi = np.stack([dat[c][i*n:(i+1)*n] for c in range(4)]).astype(np.float64)
    Vi = [np.fft.rfft(Xi[c]) for c in range(4)]
    for k in Zw:
        v = np.array([Vi[c][k] for c in range(4)])
        C = np.stack([T[:4, 0, k], T[:4, 1, k]], axis=1)
        coef, *_ = np.linalg.lstsq(C, v, rcond=None)
        Zw[k].append(coef)


for k in (8, 11, 16, 9, 15, 20):
    arr = np.array(Zw[k])            # (nwin, 2) 系数
    med = np.median(arr, axis=0)
    mean = arr.mean(axis=0)
    print('bin%-3d: median Z=(%.4f%+.4f, %.4f%+.4f)  mean Z=(%.4f%+.4f, %.4f%+.4f)'
          % (k, med[0].real, med[0].imag, med[1].real, med[1].imag,
             mean[0].real, mean[0].imag, mean[1].real, mean[1].imag))

# 理论 4 通道谱（无噪声理想渲染）在同一 bin 的形状对照
shutil.rmtree(out, ignore_errors=True)
