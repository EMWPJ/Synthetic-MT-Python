# -*- coding: utf-8 -*-
"""终局诊断：显式逐元素加权 LSQ（零推导歧义）+ 显式互谱公式对照。"""
import math
import os
import numpy as np
from synthts import calib
from synthts.model import load_site
from synthts.cli import nature_magnetic_amplitude

MU0 = 4e-7 * math.pi
base = 'lt14d'
model = base + '/MTA_dense.XMLSITE'
nfft = 1 << 16
hop = 1 << 15
site = load_site(model)
af = sorted(site.get_frequencies(False))
rmap = {x.frequency: x for x in site.fields if not x.is_noise}
sc = {}
for name in ('ex1', 'ey1', 'hx1', 'hy1', 'ex2', 'ey2', 'hx2', 'hy2'):
    v = []
    for a in af:
        row = rmap[a]
        nat = nature_magnetic_amplitude(a)
        se = nat / math.sqrt(abs(row.hx1) ** 2 + abs(row.hy1) ** 2)
        sb = nat / math.sqrt(abs(row.hx2) ** 2 + abs(row.hy2) ** 2)
        kk = 0 if name in ('ex1', 'ey1', 'hx1', 'hy1') else 1
        v.append(getattr(row, name) * (se if kk == 0 else sb))
    sc[name] = np.asarray(v)
freqs = np.fft.rfftfreq(nfft, 1.0 / 15.0)
lgb = np.log10(np.maximum(freqs, 1e-300))
lfv = np.log10(np.asarray(af, float))
Zref = {}
for k in range(1, len(freqs)):
    def ic(n):
        return complex(np.interp(lgb[k], lfv, sc[n].real),
                       np.interp(lgb[k], lfv, sc[n].imag))
    E = np.array([[ic('ex1'), ic('ex2')], [ic('ey1'), ic('ey2')]])
    H = np.array([[ic('hx1'), ic('hx2')], [ic('hy1'), ic('hy2')]])
    Zref[k] = E @ np.linalg.inv(H)

from synthts import mtprocess as mp

sr, dat = mp.read_ts(base + '/MTA.TS5')
X = np.stack([dat[c] for c in range(4)]).astype(np.float32)
win = np.hanning(nfft)
nwin = (X.shape[1] - nfft) // hop + 1
nb = nfft // 2 + 1
r = calib.responds_on_grid(freqs)

dr, dp, res = [], [], []
for k in range(1, nb - 1):
    f = freqs[k]
    if not (0.09 <= f <= 0.35) or k not in Zref:
        continue
    # 跨窗样本（该 bin 的标定后复谱）
    hx = ch = None
    hx = np.array([chx for chx in []])
    samples_h = []
    samples_e = []
    for i in range(nwin):
        seg = X[:, i * hop:i * hop + nfft] * win
        V = np.fft.rfft(seg, axis=1)
        h = V[2:4, k] * r[2][k] / MU0
        e = V[0:2, k] * r[0][k]
        samples_h.append(h)
        samples_e.append(e)
    A = np.array(samples_h)          # (N,2)
    B = np.array(samples_e)          # (N,2)
    Wd = np.abs(A).sum(axis=1) ** 2  # 亮窗权重
    # 加权 LSQ：min Σ w|e_c − z_c·h|²
    Zm = np.zeros((2, 2), dtype=complex)
    ATA = (A.conj().T * Wd) @ A
    for ci in range(2):
        ATb = (A.conj().T * Wd) @ B[:, ci]
        Zm[ci] = np.linalg.solve(ATA, ATb)
    zt = Zref[k]
    e1 = abs(abs(Zm[0, 1]) ** 2 - abs(zt[0, 1]) ** 2) / abs(zt[0, 1]) ** 2
    e2 = abs(math.degrees(np.angle(Zm[0, 1] / zt[0, 1])))
    dr.append(e1)
    dp.append(e2)
    res.append((f, e1))

dr, dp = np.array(dr), np.array(dp)
print('显式加权 LSQ（14.5d, nfft=65536, nwin=%d, n=%d）:' % (nwin, len(dr)))
print('  rho_a max=%.4f p90=%.4f median=%.4f' % (
    dr.max(), np.percentile(dr, 90), np.median(dr)))
print('  phi   max=%.3f p90=%.3f median=%.3f' % (
    dp.max(), np.percentile(dp, 90), np.median(dp)))
worst = sorted(res, key=lambda t: -t[1])[:5]
print('  worst5:', [('%.4g' % f, '%.3f' % e) for f, e in worst])
