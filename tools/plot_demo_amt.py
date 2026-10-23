# -*- coding: utf-8 -*-
"""AMT mode paper-style figures (Wang et al. 2023 style, MT twin = plot_demo.py).

-mode AMT synthesis writes three bands — TS2 (24 kHz, 0.4 s records every 60 s),
TS3 (2400 Hz, 2 s records every 60 s) and the continuous TS4 (150 Hz, from
0.01 Hz up) — then this script draws four figures:

  1. fig_time_series_amt.png — channel x samplerate panel grid (rows
     Ex/Ey/Hx/Hy, columns 24 kHz/2400 Hz/150 Hz) plus a PSD panel (d) with E
     fields on the left axis and H fields on the right axis (paper Fig 5);
  2. fig_spectrogram_amt.png — per-band Hx spectrograms (L2 burst structure);
  3. fig_rho_phase_amt.png — apparent resistivity and phase vs frequency
     (log axis reversed: high frequency left); theoretical black lines,
     synthetic values as open symbols (XY red circles, YX blue triangles)
     (paper Fig 6);
  4. fig_rho_phase_period_amt.png — the same processed data on a period axis
     with per-decade error statistics.

Usage (repo root; needs the plotting extra: pip install -e ".[plot]"):
  python tools/plot_demo_amt.py
"""
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from synthts import calib, mtprocess

calib.SMOOTH_RESPONDS = True   # 渲染与处理同一进程内同用稠密 CTS 平滑插值

from synthts.cli import main as synth_main

MU0 = 4e-7 * math.pi

OUT_DIR = 'plot_out_amt'
SITE = 'MTL'
RES = [100.0, 10.0, 1000.0]       # layer resistivities (Ohm m), last = half-space
THICK = [300.0, 3000.0]           # layer thicknesses (m)
# 锚点 0.01 ~ 8000 Hz、6 点/倍频程（AMT 三档合成后的处理频带）
_lo, _hi, _per = math.log10(0.01), math.log10(8000.0), 6.0
ANCHORS = [10.0 ** (_lo + i / _per) for i in range(int((_hi - _lo) * _per) + 1)]
DENSE, DUR, SEED, SCALE = 10.0, 3600.0, 42, 10.0
H0 = 0.02                         # source magnetic field amplitude (A/m)

CHANNELS = ('Ex', 'Ey', 'Hx', 'Hy')
# AMT 三档采样率族：TS 档、图例文字、灰度、x 轴标签
FAMILIES = (
    ('24 kHz', 'TS2', '0.0', 'Time (s)'),
    ('2400 Hz', 'TS3', '0.35', 'Time (s)'),
    ('150 Hz', 'TS4', '0.65', 'Time (s)'),
)
CH_LS = ('-', '--', ':', '-.')    # 通道线型（族用灰度区分）

# 逐 bin 处理分带：(band, nfft, f_lo, f_hi, hop)
#   TS2/TS3 为窗口档（0.4 s/2 s 记录），hop=nfft 逐记录一窗不跨记录缝；
#   TS4 连续档 50% 重叠；nfft=75000 使 0.01 Hz 恰落整数 bin（5×150/75000）
PROC_BANDS = [
    ('TS2', 9600, 500.0, 8000.0, 9600),
    ('TS3', 4800, 7.5, 320.0, 4800),
    ('TS4', 75000, 0.01, 6.0, 37500),
]


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
        lines.append('    <Hz1>0,0</Hz1><Hz2>0,0</Hz2>')
        lines.append('  </Fields>')
    lines.append('</Site>')
    Path(path).write_text(chr(10).join(lines) + chr(10), encoding='utf-8')


def _band_scale(band, f_rep):
    """代表频点的标定换算（时间序列/PSD 显示物理单位：E=mV/km，H=nT）。"""
    r = calib.compute_responds([f_rep])
    return [r[c][0] * (1e3 if c < 2 else 1e9) for c in range(4)]


def _family_psd(out_dir, band, grid, f_rep):
    """一个采样率族的四通道 PSD（物理单位，标定同 mtprocess）。

    E: count --responds--> V/m -> mV/km (x1e3)；H: count --responds--> A/m
    -> nT (x MU0 x 1e9)。PSD = 2<|X|^2>/(fs*U)，U=sum(w^2)（加窗周期图）。
    """
    sr, data = mtprocess.read_ts(os.path.join(out_dir, '%s.%s' % (SITE, band)))
    fs = float(sr)
    scale = _band_scale(band, f_rep)
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
            psd = 2.0 * np.mean(np.abs(x * scale[c]) ** 2) / (fs * u)
            out[c][0].append(f)
            out[c][1].append(psd)
    return {c: (np.array(v[0]), np.array(v[1])) for c, v in out.items()}


# 各档时间序列面板与代表频点
_COLS = [
    ('TS2', 9600, 1.0, '(a) Time Series (24 kHz)', 'Time (s)', 1000.0),
    ('TS3', 4800, 1.0, '(b) Time Series (2400 Hz)', 'Time (s)', 100.0),
    ('TS4', 22500, 1.0, '(c) Time Series (150 Hz)', 'Time (s)', 1.0),
]


def plot_timeseries(out_dir, dense_model, png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid = [f for f in ANCHORS]
    psds = {band: _family_psd(out_dir, band, grid, f_rep)
            for band, _, _, _, _, f_rep in _COLS}

    fig = plt.figure(figsize=(10.5, 6.6))
    gs = fig.add_gridspec(4, 5, hspace=0.45, wspace=0.42)
    band_scale = {band: _band_scale(band, f_rep)
                  for band, _, _, _, _, f_rep in _COLS}
    for c, ch in enumerate(CHANNELS):
        for k, (band, n_show, t_scale, title, xlab, _) in enumerate(_COLS):
            ax = fig.add_subplot(gs[c, k])
            sr, data = mtprocess.read_ts(os.path.join(out_dir, '%s.%s' % (SITE, band)))
            fs = float(sr)
            n = min(n_show, len(data[c]))
            t = np.arange(n) / fs / t_scale
            ax.plot(t, data[c][:n] * band_scale[band][c], lw=0.5, color='k')
            if k == 0:
                ax.set_ylabel('%s (%s)' % (ch, 'mV/km' if c < 2 else 'nT'))
            else:
                ax.set_yticklabels([])
            if c == 3:
                ax.set_title(title, fontsize=9)
                ax.set_xlabel(xlab)
            ax.grid(True, lw=0.3, ls=':', alpha=0.5)

    # (d) PSD：E 场挂左轴、H 场挂右轴；族=灰度、通道=线型（论文 Fig 5d）
    ax_e = fig.add_subplot(gs[0:4, 3:5])
    ax_h = ax_e.twinx()
    fam_color = {lab: col for lab, _, col, _ in FAMILIES}
    for lab, band, col, _ in FAMILIES:
        for c in range(4):
            f_arr, p_arr = psds[band][c]
            if not len(f_arr):
                continue
            target = ax_e if c < 2 else ax_h
            target.loglog(f_arr, p_arr, CH_LS[c], color=col, lw=1.0)
    ax_e.set_xlim(10000, 0.01)
    ax_e.set_xlabel('Frequency (Hz)')
    ax_e.set_ylabel('E Fields PSD ((mV/km)$^2$/Hz)')
    ax_h.set_ylabel('H Fields PSD (nT$^2$/Hz)')
    ax_e.set_title('(d) PSD of time series', fontsize=9)
    ax_e.grid(True, which='both', lw=0.3, ls=':', alpha=0.5)
    fam_handles = [plt.Line2D([], [], color=fam_color[lab], lw=1.2, label=lab)
                   for lab, _, _, _ in FAMILIES]
    ch_handles = [plt.Line2D([], [], color='k', ls=ls, lw=1.2, label=ch)
                  for ch, ls in zip(CHANNELS, CH_LS)]
    leg1 = ax_e.legend(handles=fam_handles, loc='upper left', fontsize=7, title='Samplerate')
    leg1.get_title().set_fontsize(7)
    ax_h.legend(handles=ch_handles, loc='lower left', fontsize=7)

    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def _stft_db(x, fs, nfft, hop):
    """numpy STFT → (freqs, times, dB 幅度)。"""
    w = np.hanning(nfft)
    nfrm = max((len(x) - nfft) // hop + 1, 1)
    S = np.empty((nfft // 2 + 1, nfrm), dtype=np.float64)
    for i in range(nfrm):
        S[:, i] = np.abs(np.fft.rfft(x[i * hop:i * hop + nfft] * w))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    times = np.arange(nfrm) * hop / fs
    db = 20.0 * np.log10(S / max(S.max(), 1e-300) + 1e-12)
    return freqs, times, db


def plot_spectrogram(out_dir, png):
    """三档采样率的 Hx 时频谱（L2 分带突发的可视化：能量带随时间起落）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = [  # (band, nfft, hop, f_lo, f_hi, title, x_lab)
        ('TS2', 1 << 11, 1 << 9, 400.0, 12000.0,
         '(a) Hx spectrogram, 24 kHz', 'Time (s, record-concat)'),
        ('TS3', 1 << 11, 1 << 9, 7.5, 1200.0,
         '(b) Hx spectrogram, 2400 Hz', 'Time (s, record-concat)'),
        ('TS4', 1 << 12, 1 << 10, 0.01, 75.0,
         '(c) Hx spectrogram, 150 Hz', 'Time (s)'),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for ax, (band, nfft, hop, f_lo, f_hi, title, x_lab) in zip(axes, panels):
        sr, data = mtprocess.read_ts(os.path.join(out_dir, '%s.%s' % (SITE, band)))
        fs = float(sr)
        x = data[2].astype(np.float64)                      # Hx
        nfft = min(nfft, len(x))
        freqs, times, db = _stft_db(x, fs, nfft, max(hop, nfft // 4))
        m = (freqs >= f_lo) & (freqs <= f_hi)
        im = ax.pcolormesh(times, freqs[m], db[m], shading='auto',
                           cmap='magma', vmin=-50.0, vmax=0.0)
        ax.set_yscale('log')
        ax.set_ylim(f_lo, f_hi)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(x_lab)
        ax.set_ylabel('Frequency (Hz)')
        fig.colorbar(im, ax=ax, label='dB', pad=0.01)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def _fold_phase(deg):
    """MT 惯例：相位折叠到 (-90, 90]（YX 与 XY 相差 180°）。"""
    return (deg + 90.0) % 180.0 - 90.0


def _plot_rho_phase_ax(ax1, ax2, rows):
    """ρa/相位 双面板（频率轴；理论黑线 + 合成开符号）。"""
    freqs = np.array([r[0] for r in rows])
    w = 2.0 * math.pi * freqs
    comp = {  # (estimate, theory) per component, XY = [0,1], YX = [1,0]
        'XY': ([r[1][0, 1] for r in rows], [r[2][0, 1] for r in rows]),
        'YX': ([r[1][1, 0] for r in rows], [r[2][1, 0] for r in rows]),
    }
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

    ax1.set_ylim(1, 3000)
    ax1.set_ylabel(r'$\rho_a$ ($\Omega\cdot$m)')
    ax1.legend(fontsize=8, ncol=2)
    ax2.set_ylim(0, 90)
    ax2.set_yticks([0, 45, 90])
    ax2.set_ylabel(r'$\phi$ ($^\circ$)')
    ax2.set_xlabel('Frequency (Hz)')
    for ax in (ax1, ax2):
        ax.set_xlim(10000, 0.008)    # 频率轴反向：高频在左（MT 惯例）
        ax.grid(True, which='both', lw=0.3, ls=':', alpha=0.5)


def _plot_on_axis(rows, png, period_axis, suptitle):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True)
    if period_axis:
        # 周期轴：以 1/f 作 x 重画（短周期在左 = 高频在左）
        freqs = np.array([r[0] for r in rows])
        w = 2.0 * math.pi * freqs
        per = 1.0 / freqs
        comp = {
            'XY': ([r[1][0, 1] for r in rows], [r[2][0, 1] for r in rows]),
            'YX': ([r[1][1, 0] for r in rows], [r[2][1, 0] for r in rows]),
        }
        sym = {'XY': ('o', 'red'), 'YX': ('^', 'blue')}
        for name, (est, th) in comp.items():
            m, col = sym[name]
            rho_t = np.abs(np.array(th)) ** 2 / (MU0 * w)
            ph_t = _fold_phase(np.degrees(np.angle(np.array(th))))
            ls = '-' if name == 'XY' else '--'
            ax1.loglog(per, rho_t, 'k' + ls, lw=1.0, label='Theoretical ' + name)
            ax2.semilogx(per, ph_t, 'k' + ls, lw=1.0)
            rho_e = np.abs(np.array(est)) ** 2 / (MU0 * w)
            ph_e = _fold_phase(np.degrees(np.angle(np.array(est))))
            ax1.plot(per, rho_e, m, mfc='none', mec=col, ms=4.5,
                     label='Synthetic ' + name)
            ax2.plot(per, ph_e, m, mfc='none', mec=col, ms=4.5)
        ax1.set_ylim(1, 3000)
        ax1.set_ylabel(r'$\rho_a$ ($\Omega\cdot$m)')
        ax1.legend(fontsize=8, ncol=2)
        ax2.set_ylim(0, 90)
        ax2.set_yticks([0, 45, 90])
        ax2.set_ylabel(r'$\phi$ ($^\circ$)')
        ax2.set_xlabel('Period (s)')
        for ax in (ax1, ax2):
            ax.set_xlim(100.0, 1.0e-4)   # 短周期在左（高频在左）
            ax.grid(True, which='both', lw=0.3, ls=':', alpha=0.5)
    else:
        _plot_rho_phase_ax(ax1, ax2, rows)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
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
    print('=== AMT 合成层状模型 %s (rho=%s, h=%s, %d 锚点 %.4g~%.4g Hz, dense=%g/倍频程) ==='
          % (SITE, RES, THICK, len(ANCHORS), ANCHORS[0], ANCHORS[-1], DENSE))
    rc = synth_main(['-model', model, '-mode', 'AMT', '-out', out_dir,
                     '-dur', str(DUR), '-seed', str(SEED),
                     '-scale', str(SCALE), '-dense', str(DENSE)])
    assert rc == 0, 'synthesis failed'
    dense_model = os.path.join(out_dir, '%s_dense.XMLSITE' % SITE)

    plot_timeseries(out_dir, dense_model, os.path.join(out_dir, 'fig_time_series_amt.png'))
    print('  figure: %s' % os.path.join(out_dir, 'fig_time_series_amt.png'))

    plot_spectrogram(out_dir, os.path.join(out_dir, 'fig_spectrogram_amt.png'))
    print('  figure: %s' % os.path.join(out_dir, 'fig_spectrogram_amt.png'))

    # 逐 bin 处理（渲染同构理论参考，SMOOTH_RESPONDS 与渲染同用）
    rows = []
    for band, nfft, f_lo, f_hi, hop in PROC_BANDS:
        got = mtprocess.process_fft_bins(dense_model, out_dir, band=band,
                                         nfft=nfft, site_name=SITE,
                                         f_min=f_lo, f_max=f_hi, hop=hop,
                                         window='hann')
        print('  %s nfft=%d: %d bins (%g-%g Hz)'
              % (band, nfft, len(got), f_lo, f_hi))
        rows += got
    rows.sort(key=lambda r: r[0])

    # 锚点最近 bin 输出（对数等间隔，去重）
    allbins = np.array([r[0] for r in rows])
    keep, seen = [], set()
    for a in ANCHORS:
        k = int(np.argmin(np.abs(allbins - a)))
        if k not in seen:
            seen.add(k)
            keep.append(rows[k])

    _plot_on_axis(keep, os.path.join(out_dir, 'fig_rho_phase_amt.png'),
                  period_axis=False, suptitle=None)
    print('  figure: %s' % os.path.join(out_dir, 'fig_rho_phase_amt.png'))
    _plot_on_axis(keep, os.path.join(out_dir, 'fig_rho_phase_period_amt.png'),
                  period_axis=True,
                  suptitle='AMT synthetic vs theory (dense CTS smooth, %d s record)' % int(DUR))
    print('  figure: %s' % os.path.join(out_dir, 'fig_rho_phase_period_amt.png'))

    # 分十倍程误差（Zxy）
    print('=== 分十倍程误差（Zxy, %d 频点）===' % len(keep))
    for lo in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        sel = [r for r in keep if lo <= r[0] < lo * 10.0]
        if not sel:
            continue
        e = [abs(r[1][0, 1] - r[2][0, 1]) / max(abs(r[2][0, 1]), 1e-30) for r in sel]
        ph = [math.degrees(math.atan2(r[1][0, 1].imag, r[1][0, 1].real)
                           - math.atan2(r[2][0, 1].imag, r[2][0, 1].real)) for r in sel]
        print('  %8g-%-8g Hz: n=%3d  rhoa med=%.3e p90=%.3e  phi_med=%.3f deg'
              % (lo, lo * 10, len(sel), float(np.median(e)),
                 float(np.percentile(e, 90)),
                 float(np.median(np.abs(ph)))))
    errs = [abs(r[1][0, 1] - r[2][0, 1]) / max(abs(r[2][0, 1]), 1e-30) for r in keep]
    print('=== 总计: %d 频点, Zxy 相对误差 median=%.3e max=%.3e ==='
          % (len(errs), float(np.median(errs)), max(errs)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
