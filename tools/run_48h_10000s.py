# -*- coding: utf-8 -*-
"""原始文献式 48h 合成 + 全频段处理（320 Hz ~ 10000 s 周期）。

模型：层状 H 型（100/10/1000 Ohm m），锚点半十倍程 320 Hz ~ 1e-4 Hz。
处理分带（逐 bin、Hann 块窗、hop 50%）：
  TS3 4800   : 75-320 Hz
  TS4 1200   : 3.75-72 Hz
  TS5 8192   : 0.075-3.75 Hz
  TS5 15000  : 0.0075-0.075 Hz（bin 距使锚点近整 bin）
  TS5 150000 : 8e-5-0.0075 Hz（1e-4 Hz 恰在 bin 1；10000s 窗、33 窗）
超低频带不做台阶剔除（±2bin 在 1e-4Hz 处覆盖整个十倍程，且 CTS 响应
在超低频近平坦、近邻台阶幅度小）。
"""
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from synthts import calib, mtprocess
from synthts.cli import main as synth_main
from synthts.model import densify_frequencies

MU0 = 4e-7 * math.pi
OUT_DIR = 'run48h'
SITE = 'MTL'
RES = [100.0, 10.0, 1000.0]
THICK = [300.0, 3000.0]
ANCHORS_SPARSE = [320.0, 100.0, 32.0, 10.0, 3.2, 1.0, 0.32, 0.1, 0.032,
                  0.01, 0.0032, 0.001, 3.2e-4, 1e-4]
ANCHORS = densify_frequencies(ANCHORS_SPARSE, 4.0)   # 对数等间隔 4 点/倍频程
DUR = 48 * 3600


def z1d(freq):
    w = 2j * math.pi * freq
    z = np.sqrt(w * MU0 * RES[-1])
    for rho, h in zip(reversed(RES[:-1]), reversed(THICK)):
        eta = np.sqrt(w * MU0 * rho)
        th = np.tanh(eta * h / rho)
        z = (z + eta * th) / (1.0 + z * th / eta)
    return z


def build_model(path):
    th = math.radians(30.0)
    c, s = math.cos(th), math.sin(th)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<Site name="%s" X="100.5" Y="30.2">' % SITE,
             '  <Synthetic>False</Synthetic>']
    for f in ANCHORS:
        z = z1d(f)
        zmat = np.array([[0.0, z], [-z, 0.0]])
        h1 = np.array([complex(c, 0.0), complex(s, 0.0)]) * 0.02
        h2 = np.array([complex(-s, 0.1 * c), complex(c, 0.1 * s)]) * 0.02
        e1 = zmat @ h1
        e2 = zmat @ h2
        lines.append('  <Fields frequency="%r" isNoise="False">' % f)
        lines.append('    <Zxx>0,0</Zxx><Zxy>0,0</Zxy><Zyx>0,0</Zyx>'
                     '<Zyy>0,0</Zyy><Tzx>0,0</Tzx><Tzy>0,0</Tzy>')
        for tag, v in (('Ex1', e1[0]), ('Ey1', e1[1]), ('Hx1', h1[0]),
                       ('Hy1', h1[1]), ('Ex2', e2[0]), ('Ey2', e2[1]),
                       ('Hx2', h2[0]), ('Hy2', h2[1])):
            lines.append('    <%s>%r,%r</%s>'
                         % (tag, float(v.real), float(v.imag), tag))
        lines.append('    <Hz1>1e-6,0</Hz1><Hz2>1e-6,0</Hz2>')
        lines.append('  </Fields>')
    lines.append('</Site>')
    Path(path).write_text(chr(10).join(lines) + chr(10), encoding='utf-8')


def main():
    out = OUT_DIR
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    model = os.path.join(out, 'mtl48.XMLSITE')
    build_model(model)
    print('=== 48h 合成（%d 锚点对数等间隔 %g~%g Hz, 天然源）==='
          % (len(ANCHORS), ANCHORS[-1], ANCHORS[0]))
    rc = synth_main(['-model', model, '-out', out, '-dur', str(DUR),
                     '-seed', '42', '-scale', '10', '-dense', '0',
                     '-source', 'noise'])
    assert rc == 0, 'synthesis failed'
    dense_model = model          # -dense 0：模型即处理网格（锚点=网格点）

    # 分带：(band, nfft, f_lo, f_hi, 台阶剔除, hop, 窗)
    # 分段档(TS3/TS4)必须 hop=nfft：窗=整段（跨段时间缝的拼接窗经半窗
    # Dirichlet 泄漏混入邻 bin 的 Z，实测 −15%）。超低频用 rect+大 nfft：
    # Hann ±1bin 在 1e-4Hz 处混合 ±50% 的 Z（重尾），rect 整数 bin 零泄漏。
    # 每档最短窗 >4 倍周期（f_min 处校验）：
    #   TS3 2s/75Hz=150 | TS4 8s/3.75Hz=30 | TS5 546s/0.075Hz=41
    #   TS5 4000s/0.0075Hz=30 | TS5 5000s/1e-3Hz=5 | TS5 50000s/1e-4Hz=5
    bands = [
        ('TS3', 4800, 75.0, 320.0, True, 4800, 'hann'),
        ('TS4', 1200, 3.75, 72.0, True, 1200, 'hann'),
        ('TS5', 8192, 0.075, 3.75, True, 4096, 'hann'),
        ('TS5', 60000, 0.0075, 0.075, True, 30000, 'hann'),
        ('TS5', 75000, 0.001, 0.0075, False, 37500, 'hann'),
        ('TS5', 750000, 1e-4, 0.001, False, 375000, 'hann'),
    ]
    rows = []
    for band, nfft, flo, fhi, excl, hop, win in bands:
        r = mtprocess.process_fft_bins(dense_model, out, band=band, nfft=nfft,
                                       site_name=SITE, f_min=flo, f_max=fhi,
                                       hop=hop, window=win,
                                       exclude_cal_steps=excl)
        print('  %s nfft=%d %s: %d bins (%g-%g Hz)'
              % (band, nfft, win, len(r), flo, fhi))
        rows.extend(r)
    rows.sort(key=lambda t: t[0])

    # 输出对数等间隔化：每个模型锚点取最近的处理 bin（频点=锚点网格）
    picked = []
    for a in ANCHORS:
        j = min(range(len(rows)), key=lambda i: abs(math.log(rows[i][0] / a)))
        picked.append(rows[j])
    rows = picked
    print('对数等间隔输出: %d 频点' % len(rows))

    fa = np.array([r[0] for r in rows])
    ea = np.array([abs(abs(r[1][0, 1]) / abs(r[2][0, 1]) - 1) for r in rows])
    pa = np.array([math.degrees(math.atan2(r[1][0, 1].imag, r[1][0, 1].real)
                                - math.atan2(r[2][0, 1].imag, r[2][0, 1].real))
                   for r in rows])
    print('\n=== 分十倍程误差（Zxy 幅度）===')
    for lo, hi in ((320, 10), (10, 1), (1, 0.1), (0.1, 0.01), (0.01, 1e-4)):
        m = (fa <= lo) & (fa >= hi)
        if m.sum():
            print('  %7.3g-%-7.3g Hz: n=%3d  rhoa med=%.3e p90=%.3e  phi_med=%.3f deg'
                  % (lo, hi, m.sum(), np.median(ea[m]),
                     np.percentile(ea[m], 90), np.median(np.abs(pa[m]))))

    # ── 绘图：ρa/相位 vs 周期（10000s 在左，MT 惯例）──────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    T = 1.0 / fa
    rho_t = np.array([abs(r[2][0, 1]) ** 2 / (MU0 * 2 * math.pi * r[0]) for r in rows])
    ph_t = np.array([(math.degrees(math.atan2(r[2][0, 1].imag, r[2][0, 1].real))
                      + 90.0) % 180.0 - 90.0 for r in rows])
    rho_e = np.array([abs(r[1][0, 1]) ** 2 / (MU0 * 2 * math.pi * r[0]) for r in rows])
    ph_e = np.array([(math.degrees(math.atan2(r[1][0, 1].imag, r[1][0, 1].real))
                      + 90.0) % 180.0 - 90.0 for r in rows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.6), sharex=True)
    ax1.loglog(T, rho_t, 'k-', lw=1.2, label='Theoretical XY')
    ax1.loglog(T, rho_e, 'o', mfc='none', mec='red', ms=4,
               label='Synthetic XY (48h)')
    ax2.semilogx(T, ph_t, 'k-', lw=1.2)
    ax2.semilogx(T, ph_e, 'o', mfc='none', mec='red', ms=4)
    ax1.set_ylabel(r'$\rho_a$ ($\Omega\cdot$m)')
    ax1.legend(fontsize=9, loc='upper left')
    ax2.set_ylabel(r'$\phi$ ($^\circ$)')
    ax2.set_xlabel('Period (s)')
    for ax in (ax1, ax2):
        ax.set_xlim(2e4, 2e-3)
        ax.grid(True, which='both', lw=0.3, ls=':', alpha=0.5)
    ax2.set_yticks([0, 45, 90])
    fig.tight_layout()
    png = os.path.join('plot_out', 'fig48_rho_phase_period.png')
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print('figure:', png)
    return 0


if __name__ == '__main__':
    sys.exit(main())
