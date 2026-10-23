# -*- coding: utf-8 -*-
"""用 syscal_dll 生成稠密 CTS 表 → synthts/cts_dense.py。

网格 = 12 点/倍频程对数底网格（1e-4~12000 Hz）
     + 50Hz 工频陷波邻域线性加密（45-55Hz @0.1Hz，49-51Hz @0.05Hz）
     + 稀疏表全部锚点频率。

对齐：逐 band 组求 dll/稀疏表 的复数中位比例（dll 与 SysCal_V7 的
相位参考不同、幅度一致 ~0.3%）并施加到该组全部网格点；组内锚点处
比例散度作为一致性校验打印（50Hz 陷波邻域的锚点会偏离——稀疏表在
那里欠采样了陷波，正是本工具要修复的问题）。
"""
import ctypes
import io
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthts import calib                                       # noqa: E402

DLLP = os.environ.get('SYNTHTS_SYSCAL_DLL',
                      'D:/D104/MTDataPro/SynthTSCLI/syscal_dll.dll')
FX = os.environ.get('SYNTHTS_SYSCAL_FIXTURE',
                    'D:/D104/MTDataPro/tmp/synthdemo')


def band_of(f):
    return 2 if f > 400 else 3 if f > 8 else 4 if f > 0.8 else 5


def build_grid():
    g = []
    f = 1e-4
    while f < 12000.0:                    # 12 点/倍频程
        g.append(f)
        f *= 2 ** (1 / 12.0)
    g.extend(np.arange(44.0, 56.01, 0.1))     # 50Hz 陷波邻域
    g.extend(np.arange(48.5, 51.51, 0.05))
    g.extend(calib_anchors())                 # 稀疏表锚点
    return sorted(set(round(float(x), 10) for x in g))


def calib_anchors():
    fs = set()
    for tbl in (calib._MT, calib._AMT):
        for _band, rows_ in tbl.items():
            for f, _z5 in rows_:
                fs.add(f)
    return sorted(fs)


def eval_dll(freqs):
    dll = ctypes.WinDLL(DLLP)
    dll.syscal_native_load.restype = ctypes.c_void_p
    dll.syscal_native_load.argtypes = [ctypes.c_char_p] * 3 + [ctypes.POINTER(ctypes.c_int)]
    dll.syscal_native_eval.restype = ctypes.c_int
    dll.syscal_native_eval.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
    err = ctypes.c_int(0)
    paths = [os.path.join(FX, nm).encode('mbcs')
             for nm in ('MT2-041A.TBL', '4360.CLB', 'BMT50000.CLC')]
    h = dll.syscal_native_load(*paths, ctypes.byref(err))
    assert h, 'syscal_native_load failed: %#x' % err.value
    n = len(freqs)
    co = (ctypes.c_double * (n * 10))()
    cc = (ctypes.c_int * n)()
    farr = (ctypes.c_double * n)(*freqs)
    barr = (ctypes.c_int * n)(*[band_of(x) for x in freqs])
    rc = dll.syscal_native_eval(h, farr, barr, n, co, cc)
    assert rc == 0, 'syscal_native_eval rc=%d' % rc
    arr = np.array(co).reshape(n, 5, 2)
    return [arr[:, c, 0] + 1j * arr[:, c, 1] for c in range(5)]


def main():
    grid = build_grid()
    print('网格点数: %d (1e-4 ~ 12000 Hz)' % len(grid))
    raw = eval_dll(grid)
    # 变换到标定口径（counts/场单位），50Hz 零点以 |t|<=1e12 封顶
    ch = []
    for c in range(5):
        t = np.array([calib.transform_cts(v) for v in raw[c]])
        t = np.where(np.abs(t) > 1e12, 1e12 * t / np.maximum(np.abs(t), 1e-300), t)
        ch.append(t)
    garr = np.array(grid)

    # 对齐：按 band 组求 dll/稀疏 的复数中位比例并施加到该组全部网格点
    for band in (2, 3, 4, 5):
        rows_s = {f: z5 for tbl in (calib._MT, calib._AMT)
                  for b, rr in tbl.items() if b == band
                  for f, z5 in rr}
        if not rows_s:
            continue
        sel_f = sorted(rows_s)
        for c in range(5):
            rr = []
            for f in sel_f:
                j = int(np.argmin(np.abs(garr - f)))
                zs_t = calib.transform_cts(rows_s[f][c])
                if abs(zs_t) > 0:
                    rr.append(ch[c][j] / zs_t)
            rra = np.array(rr)
            med = complex(float(np.median(rra.real)), float(np.median(rra.imag)))
            dev = float(np.abs(rra / med - 1).max())
            grp = (garr >= min(sel_f) * 0.98) & (garr <= max(sel_f) * 1.02)
            ch[c][grp] *= med
            if c in (0, 2, 4):
                print('  band%d ch%d: 比例 |med|=%.3e∠%.1f°  锚点散度 max=%.2f%%'
                      % (band, c, abs(med), math.degrees(np.angle(med)), 100 * dev))

    buf = io.StringIO()
    buf.write('# -*- coding: utf-8 -*-\n')
    buf.write('"""稠密 CTS 表（syscal_dll 生成，工具 tools/gen_dense_cts.py）。\n\n')
    buf.write('12 点/倍频程 + 50Hz 陷波邻域加密；已按稀疏表对齐相位参考。\n')
    buf.write('行 = (频率 Hz, band, [Ex.re Ex.im Ey.re Ey.im ... Hz.re Hz.im])\n')
    buf.write('"""\n')
    buf.write('\nDENSE_CTS = [\n')
    for i, f in enumerate(grid):
        b = band_of(f)
        z5 = [ch[c][i] for c in range(5)]
        flat = ', '.join('complex(%r, %r)' % (float(v.real), float(v.imag))
                         for z in z5 for v in (z.real, z.imag))
        buf.write('    (%r, %d, [%s]),\n' % (f, b, flat))
    buf.write(']\n')
    with io.open(os.path.join('synthts', 'cts_dense.py'), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())
    print('written synthts/cts_dense.py (%d rows)' % len(grid))


if __name__ == '__main__':
    main()
