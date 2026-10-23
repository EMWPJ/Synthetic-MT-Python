# -*- coding: utf-8 -*-
"""低频链路三方对照诊断（可参数化）。

用法: python tools/diag_lowf_chain.py <base_dir> <nfft> <hop> <win:0|1> [fmax]

  A. 逐 bin 复 LSQ 估计 Z_est（块 rfft，跨窗堆叠，标定后，完整 2×2）
  B. 渲染同构参考 Z_ref（归一锚点场 log 插值 E·H⁻¹）
  C. 解析真值（钳位区/锚点 bin：Zyx=−sqrt(2j·π·f·μ0·ρ)）
win=1 时块窗用 Hann（抑制 Dirichlet 远旁瓣的标定台阶混叠）。
"""
import math
import sys

import numpy as np

from synthts import calib
from synthts.model import load_site
from synthts.cli import nature_magnetic_amplitude

MU0 = 4e-7 * math.pi
RHO = 100.0

base = sys.argv[1] if len(sys.argv) > 1 else 'lt14d'
nfft = int(sys.argv[2]) if len(sys.argv) > 2 else 150000
hop = int(sys.argv[3]) if len(sys.argv) > 3 else nfft
win_on = len(sys.argv) > 4 and sys.argv[4] == '1'
fmax = float(sys.argv[5]) if len(sys.argv) > 5 else 0.40
wlsq = len(sys.argv) > 6 and sys.argv[6] == '1'
fmin = 0.075

model = load_site(base + '/MTA_dense.XMLSITE')
rows = {x.frequency: x for x in model.fields if not x.is_noise}
af = sorted(rows)

scaled = {}
for name in ('ex1', 'ey1', 'hx1', 'hy1', 'ex2', 'ey2', 'hx2', 'hy2'):
    v = []
    for a in af:
        row = rows[a]
        nat = nature_magnetic_amplitude(a)
        se = nat / math.sqrt(abs(row.hx1) ** 2 + abs(row.hy1) ** 2)
        sb = nat / math.sqrt(abs(row.hx2) ** 2 + abs(row.hy2) ** 2)
        k = 0 if name in ('ex1', 'ey1', 'hx1', 'hy1') else 1
        v.append(getattr(row, name) * (se if k == 0 else sb))
    scaled[name] = np.asarray(v)
lf = np.log10(np.asarray(af, float))

def z_true(f):
    return np.sqrt(2j * math.pi * f * MU0 * RHO)

import synthts.mtprocess as mp
sr, dat = mp.read_ts(base + '/MTA.TS5')
X = np.stack([dat[c] for c in range(4)]).astype(np.float32)
del dat
n = X.shape[1]
print('TS5: sr=%d n=%d dur=%.0f s  nfft=%d hop=%d win=%s' % (
    sr, n, n / sr, nfft, hop, 'hann' if win_on else 'rect'))

freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
nb = nfft // 2 + 1
nwin = (n - nfft) // hop + 1
r = calib.responds_on_grid(freqs)
lgb = np.log10(np.maximum(freqs, 1e-300))

def field_at(nm):
    return (np.interp(lgb, lf, scaled[nm].real)
            + 1j * np.interp(lgb, lf, scaled[nm].imag))
E1, E2 = field_at('ex1'), field_at('ex2')
H1, H2 = field_at('hx1'), field_at('hx2')
EY1, EY2 = field_at('ey1'), field_at('ey2')
HY1, HY2 = field_at('hy1'), field_at('hy2')
det = H1 * HY2 - H2 * HY1
Zxx_r = (E1 * HY2 - E2 * HY1) / det
Zxy_r = (E2 * H1 - E1 * H2) / det
Zyx_r = (EY1 * HY2 - EY2 * HY1) / det
Zyy_r = (EY2 * H1 - EY1 * H2) / det

ks = [k for k in range(1, nb - 1) if fmin <= freqs[k] <= fmax]
Hs = {k: [] for k in ks}
EXs = {k: [] for k in ks}
EYs = {k: [] for k in ks}
w = np.hanning(nfft) if win_on else np.ones(nfft)
for i in range(nwin):
    V = np.fft.rfft(X[:, i * hop:i * hop + nfft] * w, axis=1)
    for k in ks:
        Hs[k].append((V[2, k] * r[2][k] / MU0, V[3, k] * r[3][k] / MU0))
        EXs[k].append(V[0, k] * r[0][k])
        EYs[k].append(V[1, k] * r[1][k])

errs = []
print('%10s %7s %11s %11s %11s %9s %9s %9s %5s' % (
    'f[Hz]', 'k', 'errEst-Ref', 'errRef-True', 'errEst-True',
    'residEx%', 'phiEst-Tr', 'phiEst-Ref', 'band'))
zt_yx = -z_true(af[0])
for k in ks:
    f = freqs[k]
    hs = np.array(Hs[k])
    B = np.column_stack([EXs[k], EYs[k]])
    if wlsq:
        # 亮窗加权 LSQ：w_i = Σ|H_c,i|²（暗窗/突发谷窗降权，经典 MT 处理）
        wts = (np.abs(hs) ** 2).sum(axis=1)
        sw = np.sqrt(wts)[:, None]
        zrows, *_ = np.linalg.lstsq(hs * sw, B * sw, rcond=None)
    else:
        zrows, *_ = np.linalg.lstsq(hs, B, rcond=None)
    z_est = zrows.T
    z_ref = np.array([[Zxx_r[k], Zxy_r[k]],
                      [Zyx_r[k], Zyy_r[k]]])
    e_er = (np.abs(z_est - z_ref) / np.maximum(np.abs(z_ref), 1e-30)).max()
    m_true = abs(zt_yx)
    e_rt = abs(abs(z_ref[1, 0]) - m_true) / m_true
    e_et = abs(abs(z_est[1, 0]) - m_true) / m_true
    pred = hs @ z_ref[0]
    resid = float(np.median(np.abs(pred - B[:, 0])
                            / np.maximum(np.abs(B[:, 0]), 1e-30)))
    phi_tr = (math.degrees(np.angle(z_est[1, 0] / zt_yx))
              if f <= af[0] + 1e-9 else float('nan'))
    phi_rf = math.degrees(np.angle(z_est[1, 0] / z_ref[1, 0]))
    errs.append((f, k, e_er, e_et, resid, phi_tr, phi_rf))
    print('%10.5f %7d %11.3e %11.3e %11.3e %9.2f %9.3f %9.3f %5s' % (
        f, k, e_er, e_rt, e_et, resid * 100, phi_tr, phi_rf, 'low'))

e = np.array([x[2] for x in errs])
e.sort()
n_ = len(e)
print('SUMMARY errEst-Ref: n=%d p50=%.4f p90=%.4f p99=%.4f max=%.4f  >1%%: %d bins'
      % (n_, e[n_ // 2], e[int(n_ * 0.9)], e[int(n_ * 0.99)], e[-1],
         int((e > 0.01).sum())))
pt = np.array([x[5] for x in errs if not math.isnan(x[5])])
if len(pt):
    apt = np.abs(pt)
    apt.sort()
    print('SUMMARY phi-vs-truth (clamped bins, n=%d): p50=%.4f deg max=%.4f deg'
          % (len(pt), apt[len(apt) // 2], apt[-1]))
