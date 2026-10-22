# -*- coding: utf-8 -*-
"""实验：稠密 CTS（syscal_dll 生成）对 Z 台阶地板的改善。

步骤：12 点/倍频程 + 标准网格锚点 → syscal_dll 求值 → 替换 calib 表
→ noise 合成 + mtprocess 处理 → 与嵌入表（50 点）的 Z 误差对照。
"""
import math
import os
import shutil
import sys

import numpy as np

# syscal 夹具路径经环境变量提供；稠密分支不可用时自动跳过
DLLP = os.environ.get('SYNTHTS_SYSCAL_DLL', '')
FX = os.environ.get('SYNTHTS_SYSCAL_FIXTURE', '')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_e2e_demo as demo                                  # noqa: E402
from synthts import calib, mtprocess                        # noqa: E402
from synthts.cli import main as synth_main                  # noqa: E402


def _setup_dll():
    import ctypes
    dll = ctypes.WinDLL(DLLP)
    dll.syscal_native_load.restype = ctypes.c_void_p
    dll.syscal_native_load.argtypes = [ctypes.c_char_p] * 3 + [ctypes.POINTER(ctypes.c_int)]
    dll.syscal_native_eval.restype = ctypes.c_int
    dll.syscal_native_eval.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
    return dll


def _load_dll():
    import ctypes
    dll = _setup_dll()
    err = ctypes.c_int(0)
    paths = [os.path.join(FX, n) for n in ('MT2-041A.TBL', '4360.CLB', 'BMT50000.CLC')]
    h = dll.syscal_native_load(*[x.encode('mbcs') for x in paths], ctypes.byref(err))
    assert h, hex(err.value)
    return dll, h


def band_of(f):
    return 2 if f > 400 else 3 if f > 8 else 4 if f > 0.8 else 5


def evalf(h, freqs):
    import ctypes
    n = len(freqs)
    cf = ctypes.c_double * n
    cb = ctypes.c_int * n
    co = (ctypes.c_double * (n * 5 * 2))()
    cc = (ctypes.c_int * n)()
    dll = _load_dll.__dict__.setdefault('_dll', _setup_dll())
    farr = cf(*freqs)
    barr = cb(*[band_of(x) for x in freqs])
    rc = dll.syscal_native_eval(h, farr, barr, n, co, cc)
    assert rc == 0 and all(v == 0 for v in cc), (rc, list(cc))
    return [[complex(co[i * 10 + c * 2], co[i * 10 + c * 2 + 1]) for c in range(5)]
            for i in range(n)]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_e2e_demo as demo                                  # noqa: E402
from synthts import calib, mtprocess                        # noqa: E402
from synthts.cli import main as synth_main                  # noqa: E402
import contextlib
import io


def build_dense_table():
    _, h = _load_dll()
    # 稠密网格：0.005–1500 Hz，12 点/倍频程 + 标准网格锚点
    grid = sorted(set(
        [round(10 ** (math.log10(0.005) + i * (math.log10(1500 / 0.005)) / (
            math.log2(1500 / 0.005) * 12)), 8)
         for i in range(int(math.log2(1500 / 0.005) * 12) + 1)]
        + [r[0] for r in __import__('synthts.cts_mt', fromlist=['CTS_MT']).CTS_MT]
        + [r[0] for r in __import__('synthts.cts_amt', fromlist=['CTS_AMT']).CTS_AMT]))
    rows = evalf(h, grid)
    table = []
    for f, z5 in zip(grid, rows):
        table.append((f, band_of(f), [v for z in z5 for v in (z.real, z.imag)]))
    return table


def install(table):
    calib._MT = calib._parse(table)
    calib._AMT = calib._parse(table)      # 全频段在同一表内
    calib._GRID_CACHE.clear()


def measure(tag, dur=3600):
    out = 'cts_exp'
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    demo.build_model(os.path.join(out, 'mta.XMLSITE'), 'MTA', 100.0, 0.3, 0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        rc = synth_main(['-model', os.path.join(out, 'mta.XMLSITE'), '-out', out,
                         '-dur', str(dur), '-seed', '42', '-scale', '10',
                         '-dense', '2', '-source', 'noise'])
    assert rc == 0
    ra = mtprocess.process(os.path.join(out, 'mta.XMLSITE'), out,
                           bypass_screen=True)
    rd = mtprocess.process(os.path.join(out, 'MTA_dense.XMLSITE'), out,
                           bypass_screen=True)
    ea = np.array([r[4] for r in ra])
    ed = np.array([r[4] for r in rd])
    print('%-14s 锚点 median=%.3e max=%.3e | 全网格 median=%.3e p90=%.3e max=%.3e'
          % (tag, np.median(ea), ea.max(), np.median(ed),
             np.percentile(ed, 90), ed.max()))
    shutil.rmtree(out, ignore_errors=True)
    return ed



def build_const_z(path, freqs):
    """常数 Z 模型（Z 核平均零误差 → 残差=纯处理力学地板）。"""
    Z = complex(1.0, 1.0)
    L = ['<?xml version="1.0" encoding="UTF-8"?>', '<Site name="MZ" X="0" Y="0">',
         '  <Synthetic>False</Synthetic>']
    for f in freqs:
        L.append('  <Fields frequency="%r" isNoise="False">' % f)
        L.append('    <Zxx>0,0</Zxx><Zxy>0,0</Zxy><Zyx>0,0</Zyx><Zyy>0,0</Zyy>'
                 '<Tzx>0,0</Tzx><Tzy>0,0</Tzy>')
        for tag, v in (('Ex1', 0), ('Ey1', -Z), ('Hx1', 1), ('Hy1', 0),
                       ('Ex2', Z), ('Ey2', 0), ('Hx2', 0), ('Hy2', 1)):
            L.append('    <%s>%r,%r</%s>' % (
                tag, float(np.real(v)) if isinstance(v, complex) else float(v),
                float(np.imag(v)) if isinstance(v, complex) else 0.0, tag))
        L.append('    <Hz1>0,0</Hz1><Hz2>0,0</Hz2>')
        L.append('  </Fields>')
    L.append('</Site>')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(chr(10).join(L))


def measure_const_z(tag):
    out = 'cts_exp2'
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    build_const_z(os.path.join(out, 'mz.XMLSITE'),
                  [320.0, 100.0, 10.0, 1.0, 0.1])
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        rc = synth_main(['-model', os.path.join(out, 'mz.XMLSITE'), '-out', out,
                         '-dur', '3600', '-seed', '42', '-scale', '10',
                         '-dense', '2', '-source', 'noise',
                         '-preset', 'pure-l1'])
    assert rc == 0
    rows = mtprocess.process(os.path.join(out, 'mz.XMLSITE'), out,
                             bypass_screen=True)
    errs = [abs(r[1][i, j] - r[2][i, j]) / abs(r[2][i, j])
            for r in rows for i, j in ((0, 1), (1, 0))]
    e = np.array(errs)
    print('%-18s 常数Z非对角: median=%.3e p90=%.3e max=%.3e'
          % (tag, np.median(e), np.percentile(e, 90), e.max()))
    shutil.rmtree(out, ignore_errors=True)
    return float(np.median(e))


def main():
    ed0 = measure('嵌入表(50点)')
    cz0 = measure_const_z('嵌入表(50点)')
    if not (DLLP and os.path.isfile(DLLP) and FX and os.path.isdir(FX)):
        print('syscal fixtures unavailable - set SYNTHTS_SYSCAL_DLL / '
              'SYNTHTS_SYSCAL_FIXTURE to enable the dense branch')
        return 0
    table = build_dense_table()
    per_band = {}
    for f, b, _ in table:
        per_band[b] = per_band.get(b, 0) + 1
    print('稠密表: %d 点 %s' % (len(table), per_band))
    install(table)
    ed1 = measure('稠密表(12/oct)')
    # 稠密表 + 线性插值（底层数值平滑，插值安全）
    orig = calib.responds_on_grid

    def interp_grid(freqs):
        f = np.asarray(freqs, dtype=float)
        out = [np.empty(len(f), dtype=complex) for _ in range(5)]
        for b in (2, 3, 4, 5):
            idx = np.nonzero(np.array([band_of(x) for x in f]) == b)[0]
            if not len(idx):
                continue
            g = calib._GRID_CACHE.setdefault(b, calib._grid_for(b))
            if g is None:
                continue
            gf, tv = g
            for c in range(5):
                # 幅度/unwrap 相位插值（符号安全）
                amp = np.interp(f[idx], gf, np.abs(tv[c]))
                ph = np.unwrap(np.angle(tv[c]))
                phi = np.interp(f[idx], gf, ph)
                out[c][idx] = amp * np.exp(1j * phi)
        return out
    calib.responds_on_grid = interp_grid
    ed2 = measure('稠密表+插值')
    calib.responds_on_grid = orig
    cz2 = measure_const_z('稠密表+插值')
    print('\n改善: 嵌入->稠密最近邻 %.1fx | 嵌入->稠密+插值 %.1fx (median)'
          % (np.median(ed0) / np.median(ed1), np.median(ed0) / np.median(ed2)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
