# -*- coding: utf-8 -*-
"""场指纹插值与 CTS 标定的连续性检验图（诊断工具）。

面板：
  (a) Ex1 的 re/im：锚点（圆点）+ log10(f) 域分段线性插值（线）
  (b) |Ex1| 与相位（度）：插值复合后的连续性
  (c) Hx1 的 re/im 同 (a)
  (d) CTS 标定 |r_Hx(f)| 最近邻（真台阶）+ 模型锚点竖线（对照：不连续的
      是标定，不是场插值）
数值输出：插值连续性 max|Δ|、re/im 过零段计数、相位最大逐点跳变。

用法（仓库根目录）：python tools/interp_check.py
"""
import math
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_demo as pd                                      # noqa: E402
from synthts import calib                                   # noqa: E402
from synthts.model import load_site                         # noqa: E402
from synthts.render_fft import interp_log_complex           # noqa: E402

OUT = 'interp_out'


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    model = os.path.join(OUT, 'mtl.XMLSITE')
    pd.build_model(model)
    site = load_site(model)
    af = sorted(site.get_frequencies(False))
    rows = {f.frequency: f for f in site.fields if not f.is_noise}

    fgrid = np.logspace(math.log10(af[0]), math.log10(af[-1]), 4001)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(9, 12))

    zero_cross = 0
    for ax, name, row_idx in ((axes[0], 'Ex1', 'ex1'), (axes[2], 'Hx1', 'hx1')):
        vals = np.asarray([getattr(rows[f], row_idx) for f in af], dtype=complex)
        interp = interp_log_complex(af, vals, fgrid)
        # 数值核验：逐点连续 + 段内过零
        dmax = float(np.abs(np.diff(interp)).max()
                     / max(np.abs(interp).max(), 1e-300))
        for i in range(len(af) - 1):
            if vals[i].real * vals[i + 1].real < 0 \
                    or vals[i].imag * vals[i + 1].imag < 0:
                zero_cross += 1
        ax.semilogx(fgrid, interp.real, 'b-', lw=0.9, label='re (interp)')
        ax.semilogx(fgrid, interp.imag, 'r-', lw=0.9, label='im (interp)')
        ax.semilogx(af, vals.real, 'bo', ms=5)
        ax.semilogx(af, vals.imag, 'ro', ms=5)
        ax.set_title('(%s) %s re/im: anchors + log-f piecewise-linear '
                     '(max adjacent step %.1e)' % (
                         'a' if row_idx == 'ex1' else 'c', name, dmax),
                     fontsize=9)
        ax.set_xlabel('Frequency (Hz)')
        ax.legend(fontsize=8, loc='best')

    ax = axes[1]
    vals = np.asarray([rows[f].ex1 for f in af], dtype=complex)
    interp = interp_log_complex(af, vals, fgrid)
    amp = np.abs(interp)
    ph = np.degrees(np.angle(interp))
    ax.semilogx(fgrid, amp / amp.max(), 'k-', lw=0.9, label='|Ex1| (norm)')
    ax.semilogx(fgrid, ph, 'g-', lw=0.9, label='phase (deg)')
    dph = float(np.abs(np.diff(np.unwrap(np.angle(interp)))).max())
    ax.set_title('(b) |Ex1| and phase from interpolated re/im '
                 '(max unwrapped phase step %.2e rad = continuous)' % dph,
                 fontsize=9)
    ax.set_xlabel('Frequency (Hz)')
    ax.legend(fontsize=8, loc='best')

    ax = axes[3]
    r = calib.responds_on_grid(fgrid)
    ax.semilogx(fgrid, np.abs(r[2]), 'k-', lw=1.2,
                label='|r_Hx(f)| CTS nearest-neighbour')
    for f in af:
        ax.axvline(f, color='b', ls=':', lw=0.7)
    ax.set_yscale('log')
    ax.set_title('(d) CTS calibration |r_Hx|: the REAL steps (nearest '
                 'neighbour); dotted = model anchors', fontsize=9)
    ax.set_xlabel('Frequency (Hz)')
    ax.legend(fontsize=8, loc='best')

    fig.tight_layout()
    png = os.path.join(OUT, 'fig_interp_check.png')
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print('figure: %s' % png)
    print('anchors=%d  re/im zero-crossing segments=%d' % (len(af), zero_cross))
    return 0


if __name__ == '__main__':
    sys.exit(main())
