# -*- coding: utf-8 -*-
"""作者最小数据配方实测绘图（Fig6 论文样式：反向 log 频轴、理论黑实线）。

Fig1: 配方（4 周期窗 × 2 窗 = 8 周期数据）下 ρa/相位 vs 频率——
      逐块估计散点 + 中位/分位数符号，理论为模型锚点曲线。
Fig2: 误差随数据量 N（窗数）的收敛与两种窗长（4/8 周期）的地板。
"""
import io
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from synthts.mtprocess import theoretical_impedance

MU0 = 4e-7 * math.pi
OUT = 'plot_out'

with io.open(os.path.join(OUT, 'min_recipe_results.json'), encoding='utf-8') as fh:
    raw = json.load(fh)
noise = {}
for k, d in raw.items():
    f, wp, N = k.split('|')
    noise[(float(f), float(wp), int(N))] = {kk: np.asarray(vv) for kk, vv in d.items()}
with io.open(os.path.join(OUT, 'min_recipe_flat.json'), encoding='utf-8') as fh:
    flat_raw = json.load(fh)
flat = {}
for k, d in flat_raw.items():
    f, wp, N = k.split('|')
    flat[(float(f), float(wp), int(N))] = {kk: np.asarray(vv) for kk, vv in d.items()}

# 模型锚点理论曲线（渲染真值口径）
theory = dict(theoretical_impedance('syn24h/MTA_dense.XMLSITE'))
afs = np.array(sorted(theory))
rhoa_th = np.array([abs(theory[f][0, 1]) ** 2 / (MU0 * 2 * math.pi * f)
                    for f in afs])
ph_th = np.array([math.degrees(math.atan2(theory[f][0, 1].imag,
                                          theory[f][0, 1].real))
                  for f in afs])

tf = sorted(k[0] for k in noise if k[1] == 4.0 and k[2] == 2)

plt.rcParams.update({'font.size': 11, 'axes.grid': True,
                     'grid.linestyle': ':', 'grid.alpha': 0.6,
                     'legend.frameon': False, 'figure.dpi': 110})

# ── Fig1：配方精度 vs 频率 ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 7.8), sharex=True)
for ax in (ax1, ax2):
    ax.invert_xaxis()
    ax.set_xscale('log')

ax1.axhline(100, color='k', lw=1.0, alpha=0.7)
ax1.plot(afs, rhoa_th, 'k-', lw=1.2, label='theory (model)')
for j, f in enumerate(tf):
    r = noise[(f, 4.0, 2)]
    ax1.plot(np.full(len(r['rhoa']), f), r['rhoa'],
             '.', color='#9ecae1', ms=3, alpha=0.4)
    m, p10, p90 = (np.percentile(r['rhoa'], q) for q in (50, 10, 90))
    ax1.errorbar(f, m, yerr=[[m - p10], [p90 - m]], fmt='b^-', lw=1.4,
                 ms=7, capsize=4,
                 label='estimates: 4-period win x 2 win' if j == 0 else None)
ax1.set_yscale('log')
ax1.set_ylabel(r'apparent resistivity ($\Omega\cdot$m)')
ax1.set_ylim(40, 300)
ax1.set_title('Minimal recipe on v2.2 natural-source engine:\n'
              'data = 8 periods, window = 4 periods, 2-window LSQ '
              '(240 block starts per frequency)')
ax1.legend(loc='upper left')

ax2.axhline(45, color='k', lw=1.0, alpha=0.7)
ax2.plot(afs, ph_th, 'k-', lw=1.2, label='theory (model)')
for j, f in enumerate(tf):
    r = noise[(f, 4.0, 2)]
    ph = 45.0 + r['ph_err']
    ax2.plot(np.full(len(ph), f), ph, '.', color='#9ecae1', ms=3, alpha=0.4)
    m, p10, p90 = (np.percentile(ph, q) for q in (50, 10, 90))
    ax2.errorbar(f, m, yerr=[[m - p10], [p90 - m]], fmt='b^-', lw=1.4,
                 ms=7, capsize=4)
ax2.set_ylabel('phase (deg)')
ax2.set_xlabel('frequency (Hz)')
ax2.set_ylim(25, 65)
fig.tight_layout()
p1 = os.path.join(OUT, 'min_recipe_fig1_accuracy.png')
fig.savefig(p1)
plt.close(fig)

# ── Fig2：误差收敛与窗长地板 ───────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
colors = {0.1: 'tab:blue', 11.7877: 'tab:orange', 84.8343: 'tab:green'}
labels = {0.1: '0.1 Hz', 11.7877: '11.8 Hz', 84.8343: '84.8 Hz'}
labeled = set()
for ax, qi in ((ax1, 50), (ax2, 90)):
    stat = 'median' if qi == 50 else 'p90'
    for f in colors:
        for wp, ls in ((4.0, '-'), (8.0, '--')):
            xs, ys = [], []
            for N in (2, 4, 8, 16):
                k = (f, wp, N)
                if k not in noise:
                    continue
                xs.append(N)
                ys.append(100 * np.percentile(noise[k]['rho_err'], qi))
            lab = '%s, %g-period win' % (labels[f], wp)
            if lab in labeled:
                lab = None
            else:
                labeled.add(lab)
            ax.plot(xs, ys, ls, marker='o', color=colors[f], lw=1.4, ms=5,
                    label=lab)
    # 平坦 Z 对照（Z 与频率无关：sqrt-f 斜率混合归零）
    xs, ys, xs8, ys8 = [], [], [], []
    for N in (2, 4, 8, 16):
        if (1.0, 4.0, N) in flat:
            xs.append(N); ys.append(100 * np.percentile(flat[(1.0, 4.0, N)]['rho_err'], qi))
        if (1.0, 8.0, N) in flat:
            xs8.append(N); ys8.append(100 * np.percentile(flat[(1.0, 8.0, N)]['rho_err'], qi))
    lab = 'flat-Z model, 1 Hz, 4-period win'
    ax.plot(xs, ys, '-s', color='k', lw=1.6, ms=6, label=None if 'flat' in labeled else lab)
    labeled.add('flat')
    ax.axhline(1.0, color='r', lw=1.2, ls=':')
    ax.text(16.6, 1.0, '1%', color='r', fontsize=10, va='center')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks([2, 4, 8, 16])
    ax.set_xticklabels(['2', '4', '8', '16'])
    ax.set_xlabel('number of windows (data = 8/16/32/64 periods)')
    ax.set_title(stat + r' $\rho_a$ relative error')
ax1.set_ylabel('error (%)')
ax1.legend(loc='upper right', fontsize=9)
fig.suptitle('Convergence vs data amount: the floor scales with the Z slope\n'
             '(sqrt-f: 2-4% at 4-period win; flat-Z: 0.26% at N=2 — synthesis exact)',
             fontsize=12)
fig.tight_layout()
p2 = os.path.join(OUT, 'min_recipe_fig2_convergence.png')
fig.savefig(p2, bbox_inches='tight')
plt.close(fig)

print('saved:', p1)
print('saved:', p2)
