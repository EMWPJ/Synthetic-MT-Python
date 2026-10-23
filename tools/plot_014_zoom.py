# -*- coding: utf-8 -*-
"""14.5 天记录的 ρa/φ 处理结果 vs 理论对比图（显式加权 LSQ，最优处理链）。"""
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

dr, dp, fsl, zl = [], [], [], []
for k in range(1, nb - 1):
    f = freqs[k]
    if not (0.09 <= f <= 0.5) or k not in Zref:
        continue
    hx = ch = None
    samples_h = []
    samples_e = []
    for i in range(nwin):
        seg = X[:, i * hop:i * hop + nfft] * win
        V = np.fft.rfft(seg, axis=1)
        samples_h.append(V[2:4, k] * r[2][k] / MU0)
        samples_e.append(V[0:2, k] * r[0][k])
    A = np.array(samples_h)
    B = np.array(samples_e)
    Wd = np.abs(A).sum(axis=1) ** 2
    ATA = (A.conj().T * Wd) @ A
    zrows = []
    for ci in range(2):
        ATb = (A.conj().T * Wd) @ B[:, ci]
        zrows.append(np.linalg.solve(ATA, ATb))
    Zm = np.array(zrows)
    zt = Zref[k]
    e1 = abs(abs(Zm[0, 1]) ** 2 - abs(zt[0, 1]) ** 2) / abs(zt[0, 1]) ** 2
    e2 = abs(math.degrees(np.angle(Zm[0, 1] / zt[0, 1])))
    dr.append(e1)
    dp.append(e2)
    fsl.append(f)
    zl.append((f, Zm[0, 1], zt[0, 1]))

dr = np.array(dr)
fsa = np.array(fsl)

# ρa/φ 理论曲线（0.09-0.5Hz 密集）
fig_theory = []
wf = np.logspace(math.log10(0.08), math.log10(0.55), 200)
zth = []
for f in wf:
    zt = Zref[k] if False else None
# 从 dense 锚点插值理论曲线（渲染同构场比）
zz = []
for f in wf:
    kk = int(np.argmin(np.abs(lgb - math.log10(f))))
    zz.append(Zref[kk][0, 1])
zz = np.array(zz)
w2 = 2 * math.pi * wf
rho_th = np.abs(zz) ** 2 / (MU0 * w2)
ph_th = np.degrees(np.angle(zz))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
ax1.loglog(wf, rho_th, 'k-', lw=1.2, label='Theoretical (field-ratio)')
rho_e = np.abs(np.array([z[1] for z in zl])) ** 2 / (MU0 * np.array([2 * math.pi * z[0] for z in zl]))
ax1.loglog(fsa, rho_e, 'ro', ms=5, mfc='none', label='Processed (weighted LSQ)')
ax1.set_ylabel(r'$\rho_a$ ($\Omega\cdot$m)')
ax1.legend(fontsize=9)
ax1.grid(True, which='both', alpha=0.4)
ax2.semilogx(wf, ph_th, 'k-', lw=1.2, label='Theoretical')
ph_e = np.degrees(np.angle(np.array([z[1] for z in zl])))
ax2.semilogx(fsa, ph_e, 'ro', ms=5, mfc='none', label='Processed')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel(r'$\phi$ (deg)')
ax2.legend(fontsize=9)
ax2.grid(True, which='both', alpha=0.4)
fig.suptitle('14.5-day record, 0.09-0.5 Hz band, weighted LSQ', fontsize=11)
fig.tight_layout()
fig.savefig(png if False else os.path.join(base, 'rho_phase_zoom.png'), dpi=150)
plt.close(fig)
print('figure:', os.path.join(base, 'rho_phase_zoom.png'))
