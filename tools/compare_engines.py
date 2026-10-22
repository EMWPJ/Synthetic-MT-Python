# -*- coding: utf-8 -*-
"""算法升级前后对比图（v1 lines 引擎 vs v2 noise 引擎，同模型同 seed）。

输出 cmp_out/ 下四张对比图：
  fig_cmp_timeseries.png  — Hx 时序（行=2400/150/15 Hz，列=lines|noise）
  fig_cmp_spectrum.png    — Hx PSD（行=引擎，列=TS3/TS5 频段；nT²/Hz 物理单位）
  fig_cmp_spectrogram.png — Hx 时频谱（行=引擎，列=TS3/TS5；log 频轴 dB）
  fig_cmp_rhophase.png    — 视电阻率/相位：理论黑线 + 两引擎估计符号
并打印数字对照表（谱连续性/带突发性/Z 精度/墙钟）。

用法（在仓库根目录执行）：python tools/compare_engines.py
"""
import contextlib
import io
import math
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_demo as pd                                      # noqa: E402
from synthts import calib, mtprocess                        # noqa: E402
from synthts.cli import main as synth_main                  # noqa: E402

OUT = 'cmp_out'
MU0 = 4e-7 * math.pi


def _synth(engine):
    out = os.path.join(OUT, engine)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    model = os.path.join(out, 'mtl.XMLSITE')
    pd.build_model(model)
    args = ['-model', model, '-out', out, '-dur', str(pd.DUR),
            '-seed', str(pd.SEED), '-scale', str(pd.SCALE)]
    if engine == 'lines':
        args += ['-dense', str(pd.DENSE), '-periods', str(pd.PERIODS)]
    else:
        args += ['-dense', str(pd.DENSE)]      # 诊断模型口径一致
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = synth_main(args + ['-source', engine])
    assert rc == 0, buf.getvalue()[-400:]
    return time.time() - t0


def _hx(band, engine):
    sr, d = mtprocess.read_ts(os.path.join(OUT, engine, '%s.%s' % (pd.SITE, band)))
    return float(sr), d[2].astype(np.float64)


def fig_timeseries():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [('TS3', 2.0, 2400), ('TS4', 16.0, 150), ('TS5', 300.0, 15)]
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex='col')
    for r, (band, win_s, fs_hint) in enumerate(rows):
        for c, engine in enumerate(('lines', 'noise')):
            fs, x = _hx(band, engine)
            n = min(int(win_s * fs), len(x))
            t = np.arange(n) / fs
            axes[r, c].plot(t, x[:n], lw=0.5, color='k')
            axes[r, c].set_ylabel('Hx (counts)')
            if r == 0:
                axes[r, c].set_title('%s  (%s)' % (
                    'v1 lines' if engine == 'lines' else 'v2 noise', band),
                    fontsize=10)
            if r == 2:
                axes[r, c].set_xlabel('Time (s)')
    fig.suptitle('Hx time series: v1 lines (segmented comb) vs v2 noise '
                 '(continuous bursty)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_cmp_timeseries.png'), dpi=150)
    plt.close(fig)


def _welch_nT(x, fs, nfft):
    """Hx 计数谱 → nT²/Hz（按频率施 responds/μ0 标定折算）。"""
    w = np.hanning(nfft)
    hop = nfft // 2
    nf = max((len(x) - nfft) // hop + 1, 1)
    acc = np.empty((nfft // 2 + 1, nf))
    for i in range(nf):
        acc[:, i] = np.abs(np.fft.rfft(x[i * hop:i * hop + nfft] * w)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    psd = np.median(acc, axis=1) * 2.0 / (fs * float((w * w).sum()))
    r = calib.responds_on_grid(freqs)
    psd = psd * np.abs(r[2]) ** 2 / MU0 ** 2 * 1e18
    return freqs, psd


def fig_spectrum():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cols = [('TS3', 1 << 14, 2.0, 1200.0), ('TS5', 1 << 12, 0.02, 7.0)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for r, engine in enumerate(('lines', 'noise')):
        for c, (band, nfft, f_lo, f_hi) in enumerate(cols):
            fs, x = _hx(band, engine)
            freqs, psd = _welch_nT(x, fs, min(nfft, len(x)))
            m = (freqs >= f_lo) & (freqs <= f_hi)
            axes[r, c].loglog(freqs[m], psd[m], lw=0.7, color='k')
            axes[r, c].set_title('%s  %s  (%s)' % (
                'v1 lines' if engine == 'lines' else 'v2 noise', band,
                'comb at grid freqs' if engine == 'lines'
                else 'continuous'), fontsize=9)
            axes[r, c].set_xlabel('Frequency (Hz)')
            axes[r, c].set_ylabel('PSD (nT$^2$/Hz)')
            axes[r, c].grid(True, which='both', lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_cmp_spectrum.png'), dpi=150)
    plt.close(fig)


def fig_spectrogram():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cols = [('TS3', 1 << 13, 1 << 11, 2.0, 1200.0),
            ('TS5', 1 << 11, 1 << 9, 0.02, 7.0)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for r, engine in enumerate(('lines', 'noise')):
        for c, (band, nfft, hop, f_lo, f_hi) in enumerate(cols):
            fs, x = _hx(band, engine)
            nfft = min(nfft, len(x))
            freqs, times, db = pd._stft_db(x, fs, nfft, max(hop, nfft // 4))
            m = (freqs >= f_lo) & (freqs <= f_hi)
            im = axes[r, c].pcolormesh(times, freqs[m], db[m], shading='auto',
                                       cmap='magma', vmin=-50, vmax=0)
            axes[r, c].set_yscale('log')
            axes[r, c].set_ylim(f_lo, f_hi)
            axes[r, c].set_title('%s  %s  (%s)' % (
                'v1 lines' if engine == 'lines' else 'v2 noise', band,
                'static ridges' if engine == 'lines' else 'band bursts'),
                fontsize=9)
            axes[r, c].set_xlabel('Time (s)')
            axes[r, c].set_ylabel('Frequency (Hz)')
            fig.colorbar(im, ax=axes[r, c], label='dB', pad=0.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_cmp_spectrogram.png'), dpi=150)
    plt.close(fig)


def fig_rhophase():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    styles = {'lines': ('o', 'red', 'v1 lines'), 'noise': ('^', 'blue', 'v2 noise')}
    for engine, (mk, col, lab) in styles.items():
        model = os.path.join(OUT, engine, '%s_dense.XMLSITE' % pd.SITE)
        rows = mtprocess.process(model, os.path.join(OUT, engine),
                                 bypass_screen=engine != 'lines')
        freqs = np.array([r[0] for r in rows])
        w = 2.0 * math.pi * freqs
        zxy_t = np.array([r[2][0, 1] for r in rows])
        zxy_e = np.array([r[1][0, 1] for r in rows])
        rho_t = np.abs(zxy_t) ** 2 / (MU0 * w)
        ph_t = pd._fold_phase(np.degrees(np.angle(zxy_t)))
        if engine == 'lines':
            ax1.loglog(freqs, rho_t, 'k-', lw=1.0, label='Theory')
            ax2.semilogx(freqs, ph_t, 'k-', lw=1.0)
        ax1.loglog(freqs, np.abs(zxy_e) ** 2 / (MU0 * w), mk, mfc='none',
                   mec=col, ms=4, label=lab)
        ax2.semilogx(freqs, pd._fold_phase(np.degrees(np.angle(zxy_e))), mk,
                     mfc='none', mec=col, ms=4)
    ax1.set_ylim(1, 1000)
    ax1.set_ylabel(r'$\rho_a$ ($\Omega\cdot$m)')
    ax1.legend(fontsize=9)
    ax2.set_ylim(0, 90)
    ax2.set_yticks([0, 45, 90])
    ax2.set_ylabel(r'$\phi$ ($^\circ$)')
    ax2.set_xlabel('Frequency (Hz)')
    for ax in (ax1, ax2):
        ax.set_xlim(400, 0.01)
        ax.grid(True, which='both', lw=0.3, alpha=0.4)
    fig.suptitle('Apparent resistivity / phase: theory vs both engines', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_cmp_rhophase.png'), dpi=150)
    plt.close(fig)


def _burstiness(x, fs, f0):
    n = int(round(64 * fs / f0))
    hop = n // 2
    w = np.hanning(n)
    kern = w * np.exp(-2j * math.pi * f0 * np.arange(n) / fs)
    pows = np.array([abs(np.dot(x[s:s + n], kern)) ** 2
                     for s in range(0, len(x) - n + 1, hop)])
    return float(np.std(np.log(pows))), float(pows.max() / pows.min())


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    times = {e: _synth(e) for e in ('lines', 'noise')}

    # 数字对照表
    fs5 = {e: _hx('TS5', e) for e in ('lines', 'noise')}
    print('=== 升级前后数字对照（同模型/seed，TS5 Hx @1Hz 带） ===')
    for e in ('lines', 'noise'):
        sd, dyn = _burstiness(fs5[e][1], fs5[e][0], 1.0)
        print('  %-6s 带功率 std(lnP)=%.2f  动态范围=%.0fx  墙钟=%.2fs'
              % (e, sd, dyn, times[e]))
    print('  （平稳窄带噪声参照 std(lnP)≈1.28；v1 段内常幅 → 窗间相关，'
          '动态范围小；v2 突发包络 → 大幅起伏）')

    fig_timeseries()
    fig_spectrum()
    fig_spectrogram()
    fig_rhophase()
    for f in sorted(os.listdir(OUT)):
        if f.endswith('.png'):
            print('  figure: %s/%s' % (OUT, f))
    return 0


if __name__ == '__main__':
    sys.exit(main())
