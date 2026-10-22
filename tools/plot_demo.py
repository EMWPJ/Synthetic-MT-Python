# -*- coding: utf-8 -*-
"""Paper-style plot demo (Wang et al. 2023, Figs 5-6 style).

Synthesizes one layered-model site, then draws two figures (fixed output
directory ./plot_out under the current working directory):

  1. time_series.png — five-channel stacked traces at the three sampling rates
     2400/150/15 Hz (paper Fig 5A-C) plus the amplitude spectrum across the
     dense grid (paper Fig 5D);
  2. rho_phase.png — apparent resistivity and phase estimated from the synthetic
     time series (least squares) vs the 1D forward theory (paper Fig 6 style:
     blue circles line = theory, red dashed crosses = processed, Period axis).

Usage (repo root; needs the plotting extra: pip install -e ".[plot]"):
  python tools/plot_demo.py
"""
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from synthts import mtprocess
from synthts.cli import main as synth_main
from synthts.model import load_site

MU0 = 4e-7 * math.pi

OUT_DIR = 'plot_out'
SITE = 'MTL'                      # layered demo site (H-type curve)
RES = [100.0, 10.0, 1000.0]       # layer resistivities (Ohm m), last entry = half-space
THICK = [300.0, 3000.0]           # layer thicknesses (m)
ANCHORS = [320.0, 100.0, 32.0, 10.0, 3.2, 1.0, 0.32, 0.1, 0.032, 0.01]
DENSE, DUR, PERIODS, SEED, SCALE = 2.0, 3600.0, 64, 42, 10.0
H0 = 0.02                         # source magnetic field amplitude (A/m)


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
    """XMLSITE with two orthogonal source polarizations and E = Z(1D) · H."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<Site name="%s" X="100.5" Y="30.2">' % SITE,
             '  <Synthetic>False</Synthetic>']
    for f in ANCHORS:
        z = z1d(f)
        zmat = np.array([[0.0, z], [-z, 0.0]])
        h1 = np.array([complex(H0, 0.0), complex(0.0, 0.0)])
        h2 = np.array([complex(0.0, 0.0), complex(H0, 0.0)])
        e1, e2 = zmat @ h1, zmat @ h2
        lines.append('  <Fields frequency="%r" isNoise="False">' % f)
        lines.append('    <Zxx>0,0</Zxx><Zxy>0,0</Zxy><Zyx>0,0</Zyx><Zyy>0,0</Zyy>'
                     '<Tzx>0,0</Tzx><Tzy>0,0</Tzy>')
        for tag, v in (('Ex1', e1[0]), ('Ey1', e1[1]),
                       ('Hx1', h1[0]), ('Hy1', h1[1]),
                       ('Ex2', e2[0]), ('Ey2', e2[1]),
                       ('Hx2', h2[0]), ('Hy2', h2[1])):
            lines.append('    <%s>%s</%s>' % (tag, cstr(v), tag))
        lines.append('    <Hz1>1e-6,0</Hz1><Hz2>1e-6,0</Hz2>')
        lines.append('  </Fields>')
    lines.append('</Site>')
    Path(path).write_text(chr(10).join(lines) + chr(10), encoding='utf-8')


def _stack(ax, ts_path, fs_hint, n_show, title):
    sr, data = mtprocess.read_ts(ts_path)
    fs = sr or fs_hint
    t = np.arange(n_show) / fs
    for c, name in enumerate(('Ex', 'Ey', 'Hx', 'Hy', 'Hz')):
        x = data[c][:n_show]
        sd = float(x.std()) or 1.0
        ax.plot(t, (x - x.mean()) / sd + 3.0 * (4 - c), lw=0.6, color='k')
    ax.set_yticks([3.0 * k for k in range(5)])
    ax.set_yticklabels(['Ex', 'Ey', 'Hx', 'Hy', 'Hz'])
    ax.set_title(title)
    ax.set_xlim(0, t[-1])


def plot_timeseries(out_dir, dense_model, png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    _stack(axes[0, 0], os.path.join(out_dir, '%s.TS3' % SITE), 2400,
           2400, '(A) Time series, 2400 Hz')
    _stack(axes[0, 1], os.path.join(out_dir, '%s.TS4' % SITE), 150,
           1500, '(B) Time series, 150 Hz')
    _stack(axes[1, 0], os.path.join(out_dir, '%s.TS5' % SITE), 15,
           1500, '(C) Time series, 15 Hz')

    ax = axes[1, 1]
    freqs = load_site(dense_model).get_frequencies(False)
    for c, name in ((0, 'Ex'), (1, 'Ey'), (2, 'Hx'), (3, 'Hy')):
        amp = []
        for f in freqs:
            picked = mtprocess.pick_ts_file(f, out_dir, SITE)
            if picked is None:
                amp.append(np.nan)
                continue
            _, fs, data, _ = picked
            x = mtprocess.window_dft(data[c], fs, f)
            amp.append(float(np.median(np.abs(x))) if len(x) else np.nan)
        ax.loglog(freqs, amp, lw=1.0, label=name)
    ax.set_title('(D) Amplitude spectrum (windowed DFT)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.legend(fontsize=8)

    axes[1, 0].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def plot_rho_phase(rows, png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    freqs = np.array([r[0] for r in rows])
    per = 1.0 / freqs
    w = 2.0 * math.pi * freqs
    zxy_e = np.array([r[1][0, 1] for r in rows])
    zxy_t = np.array([r[2][0, 1] for r in rows])
    rho_e, rho_t = np.abs(zxy_e) ** 2 / (MU0 * w), np.abs(zxy_t) ** 2 / (MU0 * w)
    ph_e, ph_t = np.degrees(np.angle(zxy_e)), np.degrees(np.angle(zxy_t))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    ax1.loglog(per, rho_t, 'b-o', markersize=4, label='Theory')
    ax1.loglog(per, rho_e, 'r--x', markersize=5, label='Synthetic (LSQ)')
    ax1.set_ylabel(r'App. Resistivity ($\Omega \cdot m$)')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)
    ax2.semilogx(per, ph_t, 'b-o', markersize=4, label='Theory')
    ax2.semilogx(per, ph_e, 'r--x', markersize=5, label='Synthetic (LSQ)')
    ax2.set_xlabel('Period (s)')
    ax2.set_ylabel('Phase (deg)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
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
    print('=== 合成层状模型 %s (rho=%s, h=%s, %d 锚点 %g~%g Hz) ==='
          % (SITE, RES, THICK, len(ANCHORS), ANCHORS[-1], ANCHORS[0]))
    rc = synth_main(['-model', model, '-out', out_dir, '-dur', str(DUR),
                     '-periods', str(PERIODS), '-seed', str(SEED),
                     '-scale', str(SCALE), '-dense', str(DENSE)])
    assert rc == 0, 'synthesis failed'
    dense_model = os.path.join(out_dir, '%s_dense.XMLSITE' % SITE)

    plot_timeseries(out_dir, dense_model, os.path.join(out_dir, 'time_series.png'))
    print('  figure: %s' % os.path.join(out_dir, 'time_series.png'))

    rows = mtprocess.process(dense_model, out_dir)
    plot_rho_phase(rows, os.path.join(out_dir, 'rho_phase.png'))
    print('  figure: %s' % os.path.join(out_dir, 'rho_phase.png'))

    # 纯 1D 模型理论 Z 对角元恒为 0，process() 的逐元素相对误差在対角上失真，
    # 汇总口径改用 Zxy（与图一致）
    errs = [abs(r[1][0, 1] - r[2][0, 1]) / max(abs(r[2][0, 1]), 1e-30)
            for r in rows]
    print('=== rho/phase: %d 频点, Zxy 相对误差 median=%.3e max=%.3e ==='
          % (len(errs), float(np.median(errs)), max(errs)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
