# -*- coding: utf-8 -*-
"""双引擎对拍报告（设计 §5.5，M3 交付；CI/本地产物，stdout 输出）。

对同一模型（双测点同源批量）分别以 lines / noise 引擎合成，逐维度对拍：
锚点 Z / 多测点同源 / 全加密网格 Z / 谱特征 / 统计（noise）/ 墙钟（参考）。
判据口径见设计文档 §5.3。落盘方式：调用方重定向 stdout（如
`python tools/par_report.py > par_report.md`）。

用法（在仓库根目录执行）：python tools/par_report.py
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
import mt_e2e_demo as demo                                  # noqa: E402
import calibrate_real as cal                                # noqa: E402
from synthts import mtprocess                               # noqa: E402
from synthts.cli import main as synth_main                  # noqa: E402

DUR = '3600'
SEED = '42'


def _synth(outdir, engine):
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    demo.build_model(os.path.join(outdir, 'mta.XMLSITE'), 'MTA', 100.0, 0.3, 0.0)
    demo.build_model(os.path.join(outdir, 'mtb.XMLSITE'), 'MTB', 30.0, 0.45, 35.0)
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = synth_main(['-model', os.path.join(outdir, 'mta.XMLSITE'),
                         os.path.join(outdir, 'mtb.XMLSITE'),
                         '-out', outdir, '-scale', str(demo.SCALE),
                         '-seed', SEED, '-dur', DUR,
                         '-periods', str(demo.PERIODS),
                         '-dense', str(demo.DENSE), '-source', engine])
    assert rc == 0, buf.getvalue()[-500:]
    return time.time() - t0


def _engine_metrics(outdir, engine):
    bypass = engine != 'lines'
    m = {}
    rows_a = mtprocess.process(os.path.join(outdir, 'mta.XMLSITE'), outdir,
                               bypass_screen=bypass)
    rows_d = mtprocess.process(os.path.join(outdir, 'MTA_dense.XMLSITE'),
                               outdir, bypass_screen=bypass)
    m['anchor_max'] = max(r[4] for r in rows_a)
    errs = np.array([r[4] for r in rows_d])
    m['grid_median'] = float(np.median(errs))
    m['grid_p90'] = float(np.percentile(errs, 90))
    coh = [demo.coherence_residual('MTA', 'MTB', outdir, f)
           for f in demo.FREQS]
    m['coherence_worst'] = max(c for c in coh if c is not None)
    return m


def _spectral_metrics(outdir, engine):
    if engine == 'lines':
        vals = []
        for f_lo, f_hi in ((0.1, 1.0), (1.0, 10.0), (10.0, 100.0)):
            f_mid = math.sqrt(f_lo * f_hi)
            a_mid = demo.band_amp('MTA', outdir, f_mid)
            a_ref = demo.band_amp('MTA', outdir, f_mid * 2 ** 0.25)
            if a_mid and a_ref:
                vals.append(a_mid / a_ref)
        return {'gap_ratio_min': min(vals) if vals else float('nan')}
    grid = demo.densify_frequencies(demo.FREQS, demo.DENSE)
    vals = [demo.continuity_ratio('MTA', outdir, lo, hi, grid)
            for lo, hi in ((0.1, 1.0), (1.0, 10.0), (10.0, 100.0))]
    vals = [v for v in vals if v is not None]
    return {'cont_norm_min': min(vals) if vals else float('nan')}


def _stats_metrics(outdir):
    p = cal.calibrate(outdir, 'MTA', 1.0)
    return {'sigma_env_ln': p['sigma_env_ln'],
            'burst_median_s': p.get('burst_median_s', float('nan')),
            'pol_corr': p['pol_corr']}


def main():
    report = ['# 双引擎对拍报告（lines vs noise）',
              '',
              '模型：双测点同源批量（MTA ρ=100 / MTB ρ=30），%s s，seed=%s，'
              'dense=%g/倍频程，periods=%d' % (DUR, SEED, demo.DENSE,
                                               demo.PERIODS),
              '']
    rows = {}
    for engine in ('lines', 'noise'):
        outdir = 'par_%s' % engine
        t = _synth(outdir, engine)
        rows[engine] = {'time': t}
        rows[engine].update(_engine_metrics(outdir, engine))
        rows[engine].update(_spectral_metrics(outdir, engine))
        if engine == 'noise':
            rows[engine].update(_stats_metrics(outdir))
        shutil.rmtree(outdir, ignore_errors=True)

    report += ['| 维度 | lines | noise | 判据（noise） |',
               '|---|---|---|---|',
               '| 锚点 Z 最大相对误差 | %.2e | %.2e | <1e-1 |'
               % (rows['lines']['anchor_max'], rows['noise']['anchor_max']),
               '| 多测点同源残差 | %.2e | %.2e | <2e-2 |'
               % (rows['lines']['coherence_worst'],
                  rows['noise']['coherence_worst']),
               '| 全网格 Z median | %.2e | %.2e | <3e-2 |'
               % (rows['lines']['grid_median'], rows['noise']['grid_median']),
               '| 全网格 Z p90 | %.2e | %.2e | <6e-2 |'
               % (rows['lines']['grid_p90'], rows['noise']['grid_p90']),
               '| 谱特征（lines=间隙比 / noise=连续比） | %.3f | %.3f | noise>0.8 |'
               % (rows['lines']['gap_ratio_min'],
                  rows['noise']['cont_norm_min']),
               '| 带内包络 σ_env(ln) | — | %.3f | 报告项 |'
               % rows['noise']['sigma_env_ln'],
               '| 突发段中位 (s) | — | %.1f | 报告项 |'
               % rows['noise']['burst_median_s'],
               '| 极化相关 Hx/Hy | — | %.2f | |ρ|<0.3 |'
               % rows['noise']['pol_corr'],
               '| 墙钟（单次参考，正式基准见 cost_bench） | %.2f s | %.2f s | — |'
               % (rows['lines']['time'], rows['noise']['time']),
               '',
               '注：Z 判据口径见设计文档 §5.3（noise 的 1~2e-2 地板为 CTS 最近',
               '邻阶跃×连续谱窗核的混窗失配，加长窗不改善）；统计维度仅在',
               'noise 有定义（v1 分段常幅不构成可比统计）。',
               '']
    print(chr(10).join(report))
    ok = (rows['noise']['anchor_max'] < 1e-1
          and rows['noise']['coherence_worst'] < 2e-2
          and rows['noise']['grid_median'] < 3e-2
          and rows['noise']['grid_p90'] < 6e-2
          and rows['noise']['cont_norm_min'] > 0.8
          and abs(rows['noise']['pol_corr']) < 0.3)
    print('==== par_report: %s ====' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
