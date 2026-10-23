# -*- coding: utf-8 -*-
"""平坦 Z 对照实验：Z 与频率无关时，作者最小配方（4 周期窗×2 窗）的精度。

结论（v2.2 引擎实测）：win=4/N=2 中位 0.26-0.61%——合成算法精确；
sqrt-f 模型同配置的 3-8% 全部来自 "Z 斜率 x Hann ±1bin 泄漏"：
Hann 窗 DFT 在整数 bin 上仅 W(0)=1、W(±1)=0.5 有值（其余恰为 0），
相邻 bin 在 4 周期窗下相距 0.25*f，sqrt-f 律使相邻 bin Z 差 ~12%，
0.5x12% 的混合经 /H_k 重尾化后即观测地板；8 周期窗 bin 距减半、地板减半。
"""
import contextlib
import io
import math
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthts import calib                                          # noqa: E402
from synthts.cli import main as synth_main                         # noqa: E402
from synthts.mtprocess import theoretical_impedance                # noqa: E402
import test_minimal_recipe as tmr                                  # noqa: E402

MU0 = 4e-7 * math.pi


def build_flat_model(path, name):
    zc = np.sqrt(2j * math.pi * 0.1 * MU0 * 100.0)   # 0.1Hz 的 zxy，全频段不变
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<Site name="%s" X="100.5" Y="30.2">' % name,
             '  <Synthetic>False</Synthetic>']
    for f in (320.0, 100.0, 32.0, 10.0, 3.2, 1.0):
        z = np.array([[0.3 * zc, zc], [-zc, -0.18 * zc]])
        h1 = np.array([1.0, 0.0]) * 0.02
        h2 = np.array([0.0, 1.0]) * 0.02
        e1 = z @ h1
        e2 = z @ h2
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
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(chr(10).join(lines) + chr(10))


def main():
    tag = 'syn6h_flat'
    if os.path.isdir(tag):
        shutil.rmtree(tag)
    os.makedirs(tag)
    m = tag + '/mtflat.XMLSITE'
    build_flat_model(m, 'MTA')
    with contextlib.redirect_stdout(io.StringIO()):
        rc = synth_main(['-model', m, '-out', tag, '-dur', '21600',
                         '-seed', '42', '-scale', '10', '-dense', '2',
                         '-source', 'noise'])
    assert rc == 0, 'flat synthesis failed'
    theory = dict(theoretical_impedance(tag + '/MTA_dense.XMLSITE'))
    for band, f in (('TS5', 1.0), ('TS4', 10.0), ('TS3', 100.0)):
        if f not in theory:
            continue
        sr, data, segs = tmr.read_ts_segments(tag + '/MTA.%s' % band)
        zt = theory[f]
        r = calib.compute_responds([f])
        for wp in (4.0, 8.0):
            n = int(round(wp * sr / f))
            if n < 12 or sr / f < 4.0:
                continue
            for N in (2, 4, 8, 16):
                starts = tmr.block_starts(segs, n, N)
                if not starts:
                    continue
                if len(starts) > 240:
                    starts = starts[::int(math.ceil(len(starts) / 240))][:240]
                V = tmr.win_dfts(data, np.asarray(starts), n, N, f, sr)
                hx = V[2] * r[2][0] / MU0
                hy = V[3] * r[3][0] / MU0
                ex = V[0] * r[0][0]
                ey = V[1] * r[1][0]
                errs = []
                for b in range(len(starts)):
                    h = np.column_stack([hx[b], hy[b]])
                    B = np.column_stack([ex[b], ey[b]])
                    z = np.linalg.lstsq(h, B, rcond=None)[0].T
                    errs.append(abs(abs(z[0, 1]) / abs(zt[0, 1]) - 1))
                e = np.array(errs)
                print('f=%6g win=%g N=%2d: rhoa med=%.3e p90=%.3e'
                      % (f, wp, N, np.median(e), np.percentile(e, 90)))


if __name__ == '__main__':
    main()
