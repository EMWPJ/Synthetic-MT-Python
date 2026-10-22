# -*- coding: utf-8 -*-
"""Paper-style plot demo (Wang et al. 2023, Figs 5-6 as drawn by the author).

Synthesizes one layered-model site, then draws two figures following the
author's own GMT scripts (Figure 5.ps1 / Figure 6.ps1):

  1. fig_time_series.png — channel x samplerate panel grid (rows Ex/Ey/Hx/Hy,
     columns 2400/150/15 Hz; a 1D model has no vertical magnetic field, so Hz
     is not drawn), each panel with its own amplitude axis, plus a PSD panel
     (d) with E fields on the left axis and H fields on the right axis, one
     curve family per samplerate (paper Fig 5);
  2. fig_rho_phase.png — apparent resistivity and phase vs FREQUENCY (log axis
     reversed: high frequency left): theoretical black lines, synthetic values
     as open symbols (XY red circles, YX blue triangles) (paper Fig 6).

Usage (repo root; needs the plotting extra: pip install -e ".[plot]"):
  python tools/plot_demo.py
"""
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from synthts import calib, mtprocess
from synthts.cli import main as synth_main
from synthts.model import load_site

MU0 = 4e-7 * math.pi

OUT_DIR = 'plot_out'
SITE = 'MTL'                      # layered demo site (H-type curve)
RES = [100.0, 10.0, 1000.0]       # layer resistivities (Ohm m), last entry = half-space
THICK = [300.0, 3000.0]           # layer thicknesses (m)
# 锚点下限 0.1Hz：更低频在 dur=3600s 下每极化仅 1~2 个分段（128 周期分段长
# 6400s@0.01Hz），随机幅相无法跨段平均，会重现低频失衡
ANCHORS = [320.0, 100.0, 32.0, 10.0, 3.2, 1.0, 0.32, 0.1]
DENSE, DUR, PERIODS, SEED, SCALE = 10.0, 3600.0, 128, 42, 10.0
# 加密 10 点/倍频程（用户指定）：谱连续性更好、rho/phase 曲线更密；
# 段长同步 128 周期——加密越密，8 周期处理窗主瓣内的邻频段越多
# （10/oct 时 +/-0.32 倍频程约 6 个），散射随段长线性下降，需更长分段抵消
H0 = 0.02                         # source magnetic field amplitude (A/m)

CHANNELS = ('Ex', 'Ey', 'Hx', 'Hy')
# 论文 Fig 5 的三个采样率族：TS 档、图例文字、灰度、时间窗(s)、x 轴标签
FAMILIES = (
    ('2400 Hz', 'TS3', '0.0', 2.0, 'Time (s)'),
    ('150 Hz', 'TS4', '0.35', 16.0, 'Time (s)'),
    ('15 Hz', 'TS5', '0.65', 3600.0, 'Time (h)'),
)
CH_LS = ('-', '--', ':', '-.')    # 通道线型（族用灰度区分）


def z1d(freq):
    """1D layered surface impedance (recursive, bottom-up)."""
    w = 2j * math.pi * freq
    z = np.sqrt(w * MU0 * RES[-1])
    for rho, h in zip(reversed(RES[:-1]), reversed(THICK)):
        eta = np.sqrt(w * MU0 * rho)
        th = np.tanh(eta * h / rho)
        z = (z + eta * th) / (1.0 + z * th / eta)
    return z


def cstr(z):
    return '%r,%r' % (float(z.real), float(z.imag))


def build_model(path):
    """XMLSITE with two rotated elliptical source polarizations and E = Z(1D) · H."""
    th = math.radians(30.0)
    c, s = math.cos(th), math.sin(th)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<Site name="%s" X="100.5" Y="30.2">' % SITE,
             '  <Synthetic>False</Synthetic>']
    for f in ANCHORS:
        z = z1d(f)
        zmat = np.array([[0.0, z], [-z, 0.0]])
        h1 = np.array([complex(c, 0.0), complex(s, 0.0)]) * H0
        h2 = np.array([complex(-s, 0.0), complex(0.5 * c, 0.0)]) * H0
        e1, e2 = zmat @ h1, zmat @ h2
        lines.append('  <Fields frequency="%r" isNoise="False">' % f)
        lines.append('    <Zxx>0,0</Zxx><Zxy>0,0</Zxy><Zyx>0,0</Zyx><Zyy>0,0</Zyy>'
                     '<Tzx>0,0</Tzx><Tzy>0,0</Tzy>')
        for tag, v in (('Ex1', e1[0]), ('Ey1', e1[1]),
                       ('Hx1', h1[0]), ('Hy1', h1[1]),
                       ('Ex2', e2[0]), ('Ey2', e2[1]),
                       ('Hx2', h2[0]), ('Hy2', h2[1])):
            lines.append('    <%s>%s</%s>' % (tag, cstr(v), tag))
        # 纯 1D：垂直磁场恒为零（Hz1=Hz2=0，Tzx 不参与合成）
        lines.append('    <Hz1>0,0</Hz1><Hz2>0,0</Hz2>')
        lines.append('  </Fields>')
    lines.append('</Site>')
    Path(path).write_text(chr(10).join(lines) + chr(10), encoding='utf-8')


def _family_psd(out_dir, band, grid):
    """一个采样率族的四通道 PSD（物理单位，标定同 mtprocess）。

    E: count --responds--> V/m -> mV/km (x1e3)；H: count --responds--> A/m
    -> nT (x MU0 x 1e9)。PSD = 2<|X|^2>/(fs*U)，U=sum(w^2)（加窗周期图）。
    """
    sr, data = mtprocess.read_ts(os.path.join(out_dir, '%s.%s' % (SITE, band)))
    fs = float(sr)
    out = {c: ([], []) for c in range(4)}
    for f in grid:
        n = int(round(mtprocess.WINDOW_PERIODS * fs / f))
        if n < 8 or len(data[0]) < n:
            continue
        w = np.hanning(n)
        u = float((w * w).sum())
        r = calib.compute_responds([f])
        for c in range(4):
            x = mtprocess.window_dft(data[c], fs, f)
            if not len(x):
                continue
            scale = r[c][0] * (1e3 if c < 2 else 1e9)
            psd = 2.0 * np.mean(np.abs(x * scale) ** 2) / (fs * u)
            out[c][0].append(f)
            out[c][1].append(psd)
    return {c: (np.array(v[0]), np.array(v[1])) for c, v in out.items()}


def plot_timeseries(out_dir, dense_model, png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid = sorted(load_site(dense_model).get_frequencies(False))
    psds = {band: _family_psd(out_dir, band, grid) for _, band, _, _, _ in FAMILIES}

    fig = plt.figure(figsize=(10.5, 6.6))
    gs = fig.add_gridspec(4, 5, hspace=0.45, wspace=0.42)
    cols = [
        ('TS3', 4800, 1.0, '(a) Time Series (2400 Hz)', 'Time (s)'),
        ('TS4', 2400, 1.0, '(b) Time Series (150 Hz)', 'Time (s)'),
        ('TS5', 54015, 3600.0, '(c) Time Series (15 Hz)', 'Time (h)'),
    ]
    for c, ch in enumerate(CHANNELS):
        for k, (band, n_show, t_scale, title, xlab) in enumerate(cols):
            ax = fig.add_subplot(gs[c, k])
            sr, data = mtprocess.read_ts(os.path.join(out_dir, '%s.%s' % (SITE, band)))
            fs = float(sr)
            n = min(n_show, len(data[c]))
            t = np.arange(n) / fs / t_scale
            ax.plot(t, data[c][:n], lw=0.5, color='k')
            if k == 0:
                ax.set_ylabel('%s (count)' % ch)
            else:
                ax.set_yticklabels([])
            if c == 3:
                ax.set_title(title, fontsize=9)
                ax.set_xlabel(xlab)
            ax.grid(True, lw=0.3, ls=':', alpha=0.5)

    # (d) PSD：E 场挂左轴、H 场挂右轴；族=灰度、通道=线型（论文 Fig 5d）
    ax_e = fig.add_subplot(gs[0:4, 3:5])
    ax_h = ax_e.twinx()
    fam_color = {lab: col for lab, _, col, _, _ in FAMILIES}
    for lab, band, col, _, _ in FAMILIES:
        for c in range(4):
            f_arr, p_arr = psds[band][c]
            if not len(f_arr):
                continue
            target = ax_e if c < 2 else ax_h
            target.loglog(f_arr, p_arr, CH_LS[c], color=col, lw=1.0)
    ax_e.set_xlim(400, 0.01)
    ax_e.set_xlabel('Frequency (Hz)')
    ax_e.set_ylabel('E Fields PSD ((mV/km)$^2$/Hz)')
    ax_h.set_ylabel('H Fields PSD (nT$^2$/Hz)')
    ax_e.set_title('(d) PSD of time series', fontsize=9)
    ax_e.grid(True, which='both', lw=0.3, ls=':', alpha=0.5)
    fam_handles = [plt.Line2D([], [], color=fam_color[lab], lw=1.2, label=lab)
                   for lab, _, _, _, _ in FAMILIES]
    ch_handles = [plt.Line2D([], [], color='k', ls=ls, lw=1.2, label=ch)
                  for ch, ls in zip(CHANNELS, CH_LS)]
    leg1 = ax_e.legend(handles=fam_handles, loc='upper left', fontsize=7, title='Samplerate')
    leg1.get_title().set_fontsize(7)
    ax_h.legend(handles=ch_handles, loc='lower left', fontsize=7)

    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def _fold_phase(deg):
    """MT 惯例：相位折叠到 (-90, 90]（YX 与 XY 相差 180°）。"""
    return (deg + 90.0) % 180.0 - 90.0


def plot_rho_phase(rows, png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    freqs = np.array([r[0] for r in rows])
    w = 2.0 * math.pi * freqs
    comp = {  # (estimate, theory) per component, XY = [0,1], YX = [1,0]
        'XY': ([r[1][0, 1] for r in rows], [r[2][0, 1] for r in rows]),
        'YX': ([r[1][1, 0] for r in rows], [r[2][1, 0] for r in rows]),
    }
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True)
    sym = {'XY': ('o', 'red'), 'YX': ('^', 'blue')}
    for name, (est, th) in comp.items():
        m, col = sym[name]
        rho_t = np.abs(np.array(th)) ** 2 / (MU0 * w)
        ph_t = _fold_phase(np.degrees(np.angle(np.array(th))))
        ls = '-' if name == 'XY' else '--'
        ax1.loglog(freqs, rho_t, 'k' + ls, lw=1.0,
                   label='Theoretical ' + name)
        ax2.semilogx(freqs, ph_t, 'k' + ls, lw=1.0)
        rho_e = np.abs(np.array(est)) ** 2 / (MU0 * w)
        ph_e = _fold_phase(np.degrees(np.angle(np.array(est))))
        ax1.plot(freqs, rho_e, m, mfc='none', mec=col, ms=4.5,
                 label='Synthetic ' + name)
        ax2.plot(freqs, ph_e, m, mfc='none', mec=col, ms=4.5)

    ax1.set_ylim(1, 1000)
    ax1.set_ylabel(r'$\rho_a$ ($\Omega\cdot$m)')
    ax1.legend(fontsize=8, ncol=2)
    ax2.set_ylim(0, 90)
    ax2.set_yticks([0, 45, 90])
    ax2.set_ylabel(r'$\phi$ ($^\circ$)')
    ax2.set_xlabel('Frequency (Hz)')
    for ax in (ax1, ax2):
        ax.set_xlim(400, 0.01)          # 频率轴反向：高频在左（MT 惯例）
        ax.grid(True, which='both', lw=0.3, ls=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def main():
    out = Path(OUT_DIR)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    out_dir = str(out)

    model = os.path.join(out_dir, '%s.XMLSITE' % SITE.lower())
    build_model(model)
    print('=== 合成层状模型 %s (rho=%s, h=%s, %d 锚点 %g~%g Hz, dense=%g/倍频程) ==='
          % (SITE, RES, THICK, len(ANCHORS), ANCHORS[-1], ANCHORS[0], DENSE))
    rc = synth_main(['-model', model, '-out', out_dir, '-dur', str(DUR),
                     '-periods', str(PERIODS), '-seed', str(SEED),
                     '-scale', str(SCALE), '-dense', str(DENSE)])
    assert rc == 0, 'synthesis failed'
    dense_model = os.path.join(out_dir, '%s_dense.XMLSITE' % SITE)

    plot_timeseries(out_dir, dense_model, os.path.join(out_dir, 'fig_time_series.png'))
    print('  figure: %s' % os.path.join(out_dir, 'fig_time_series.png'))

    rows = mtprocess.process(dense_model, out_dir)
    plot_rho_phase(rows, os.path.join(out_dir, 'fig_rho_phase.png'))
    print('  figure: %s' % os.path.join(out_dir, 'fig_rho_phase.png'))

    # 纯 1D 模型理论 Z 对角元恒为 0，process() 的逐元素相对误差在対角上失真，
    # 汇总口径改用 Zxy（与图一致）
    errs = [abs(r[1][0, 1] - r[2][0, 1]) / max(abs(r[2][0, 1]), 1e-30)
            for r in rows]
    print('=== rho/phase: %d 频点, Zxy 相对误差 median=%.3e max=%.3e ==='
          % (len(errs), float(np.median(errs)), max(errs)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
