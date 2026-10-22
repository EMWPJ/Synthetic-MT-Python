# -*- coding: utf-8 -*-
"""v2 噪声源引擎闸门与合同测试（设计 §5.2 ⑤⑥ + 引擎合同）。

运行预算：默认 pytest 套件内 ~40s（合成 dur=1200×1 测点）。
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(REPO, 'examples', 'mta.XMLSITE')
sys.path.insert(0, os.path.join(REPO, 'tools'))

from synthts import bands, calib, mtprocess          # noqa: E402
from synthts.render_fft import interp_log_complex     # noqa: E402


def _run_noise(tmp_path, dur='1200', seed='42', extra=()):
    out = tmp_path / 'n'
    r = subprocess.run(
        [sys.executable, '-m', 'synthts', '-model', MODEL, '-out', str(out),
         '-dur', dur, '-seed', seed, '-scale', '10',
         '-source', 'noise', '-preset', 'pure-l1'] + list(extra),
        cwd=str(tmp_path), capture_output=True, text=True,
        encoding='utf-8', errors='replace')
    assert r.returncode == 0, r.stdout + r.stderr
    lines = [l for l in r.stdout.splitlines() if l.startswith('SYNTHTS_SUMMARY ')]
    return str(out), json.loads(lines[-1][len('SYNTHTS_SUMMARY '):])


def _run_noise_inproc(out, dur, seed, preset='natural'):
    """进程内合成（闸门⑦/标定用，避免子进程开销）。"""
    from synthts.cli import main as synth_main
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        rc = synth_main(['-model', MODEL, '-out', out, '-dur', str(dur),
                         '-seed', str(seed), '-scale', '10', '-dense', '0',
                         '-source', 'noise', '-preset', preset])
    assert rc == 0


def test_bands_power_complementarity():
    f = np.logspace(-4, 4, 4001)
    assert bands.power_complementarity_error(f) < 1e-12


def test_calib_grid_equivalence():
    """responds_on_grid 与逐点 compute_responds 全频段恒等（§4.2 合同）。"""
    freqs = np.logspace(-4, 4, 501)
    vec = calib.responds_on_grid(freqs)
    ref = calib.compute_responds(list(freqs))
    for c in range(5):
        assert np.array_equal(np.asarray(vec[c]), np.asarray(ref[c])), c


def test_noise_summary_and_determinism(tmp_path):
    out1, s1 = _run_noise(tmp_path / 'a' if False else tmp_path)
    # 摘要合同（§4.1）：新字段写全、summary_v=2
    assert s1['source'] == 'noise' and s1['preset'] == 'pure-l1'
    assert s1['source_params'] == {'alpha1': 1.0, 'alpha2': 0.0, 'alpha3': 0.0}
    assert s1['summary_v'] == 2 and s1['rng']['numpy']
    assert set(s1['n_bands']) == {'TS3', 'TS4', 'TS5'}
    assert s1['events_dropped'] == 0
    # 同 seed 字节复现（§3.4：同机同 numpy 逐字节一致）
    def digest(out):
        return {n: hashlib.sha256(open(os.path.join(out, n), 'rb').read()).hexdigest()
                for n in sorted(os.listdir(out)) if n.endswith(('.TBL', '.TS3', '.TS4', '.TS5'))}
    d1 = digest(out1)
    out2, _ = _run_noise(tmp_path, seed='42')
    assert digest(out2) == d1
    # lines 引擎显式调用：字段写全（null 占位）
    r = subprocess.run(
        [sys.executable, '-m', 'synthts', '-model', MODEL,
         '-out', str(tmp_path / 'l'), '-dur', '60', '-seed', '7',
         '-source', 'lines'],
        cwd=str(tmp_path), capture_output=True, text=True,
        encoding='utf-8', errors='replace')
    assert r.returncode == 0
    s = json.loads([l for l in r.stdout.splitlines()
                    if l.startswith('SYNTHTS_SUMMARY ')][-1][16:])
    assert s['source'] == 'lines' and s['preset'] is None \
        and s['n_bands'] is None and s['summary_v'] == 2
    # v2.0 起默认源为 noise：零参数调用即 natural 预设
    r = subprocess.run(
        [sys.executable, '-m', 'synthts', '-model', MODEL,
         '-out', str(tmp_path / 'd'), '-dur', '60', '-seed', '7'],
        cwd=str(tmp_path), capture_output=True, text=True,
        encoding='utf-8', errors='replace')
    assert r.returncode == 0
    s = json.loads([l for l in r.stdout.splitlines()
                    if l.startswith('SYNTHTS_SUMMARY ')][-1][16:])
    assert s['source'] == 'noise' and s['preset'] == 'natural'


def test_gate5_continuity(tmp_path):
    """闸门⑤：间隙探针斜率归一幅值比（连续谱判据，L1-only）。"""
    import mt_e2e_demo as demo
    out, _ = _run_noise(tmp_path)
    grid = sorted(demo.densify_frequencies(demo.FREQS, 2.0))
    ratios = []
    for f_lo, f_hi in ((0.1, 1.0), (1.0, 10.0), (10.0, 100.0)):
        r = demo.continuity_ratio('MTA', out, f_lo, f_hi, grid)
        if r is not None:
            ratios.append(r)
    assert len(ratios) >= 2, 'probe bands missing'
    assert min(ratios) > 0.8, ratios


def test_gate6_psd_shape(tmp_path):
    """闸门⑥：H 通道 PSD 与 |C_H|² 参考（同插值实现）带内吻合 ≤1dB 量级。"""
    out, _ = _run_noise(tmp_path)
    dense_model = os.path.join(out, 'MTA_dense.XMLSITE')
    from synthts.model import load_site
    site = load_site(dense_model)
    af = sorted(site.get_frequencies(False))
    rows = {f.frequency: f for f in site.fields if not f.is_noise}
    probs = [af[i] * math.sqrt(af[i + 1] / af[i])
             for i in range(0, len(af) - 1, 3) if 0.1 < af[i] < 100]
    # 参考 = 渲染同构：锚点处做 nature 归一（cli 预处理语义），再插值到 f
    from synthts.cli import nature_magnetic_amplitude
    scaled = {}
    for name in ('hx1', 'hy1', 'hx2', 'hy2'):
        scaled[name] = []
    for a in af:
        row = rows[a]
        nat = nature_magnetic_amplitude(a)
        se = nat / math.sqrt(abs(row.hx1) ** 2 + abs(row.hy1) ** 2)
        sb = nat / math.sqrt(abs(row.hx2) ** 2 + abs(row.hy2) ** 2)
        scaled['hx1'].append(row.hx1 * se)
        scaled['hy1'].append(row.hy1 * se)
        scaled['hx2'].append(row.hx2 * sb)
        scaled['hy2'].append(row.hy2 * sb)
    for name in scaled:
        scaled[name] = np.asarray(scaled[name])
    d10lg, k_eff = [], 0
    for f in probs:
        picked = mtprocess.pick_ts_file(f, out, 'MTA')
        if picked is None:
            continue
        _, fs, data, nwin = picked
        x = mtprocess.window_dft(data[2], fs, f)
        if len(x) < 8:
            continue
        n = int(round(mtprocess.WINDOW_PERIODS * fs / f))   # 窗长随频率变
        u = float((np.hanning(n) ** 2).sum())               # 归一化（PSD 口径）
        r = calib.compute_responds([f])
        phys = x * r[2][0] / (4e-7 * math.pi)               # ADC -> A/m
        est = float(np.median(np.abs(phys) ** 2)) / (fs * u)
        # 参考 = 锚点 nature 归一后场（渲染同构），再插值到 f
        hx1 = interp_log_complex(af, scaled['hx1'], [f])[0]
        hx2 = interp_log_complex(af, scaled['hx2'], [f])[0]
        ref = abs(hx1) ** 2 + abs(hx2) ** 2
        d10lg.append((est, ref))
        k_eff = max(k_eff, int(0.75 * len(x)))
    est_a = np.array([a for a, _ in d10lg])
    ref_a = np.array([b for _, b in d10lg])
    scale = np.median(est_a / ref_a)                    # 单一比例因子（-scale 等）
    dev = np.abs(10.0 * np.log10(est_a / (scale * ref_a)))
    thr = max(1.0, 3 * 4.34 / math.sqrt(max(k_eff, 1)))
    assert float(np.median(dev)) < thr, (float(np.median(dev)), thr)
    assert float(np.percentile(dev, 90)) < 3.0, float(np.percentile(dev, 90))


def test_screen_bypass_on_noise(tmp_path):
    """噪声源下半窗筛查通过率极低（~4%）且 bypass_screen 不破坏 LSQ（§2.4）。"""
    out, _ = _run_noise(tmp_path, dur='900')
    f = 1.0
    picked = mtprocess.pick_ts_file(f, out, 'MTA')
    assert picked is not None
    _, fs, data, _ = picked
    mask = mtprocess.screen_windows([data[0]], fs, f)
    assert mask.sum() / max(len(mask), 1) < 0.15        # 文档化的不相容性
    z_bp = mtprocess.estimate_impedance_at(f, out, 'MTA', bypass_screen=True)
    z_df = mtprocess.estimate_impedance_at(f, out, 'MTA')
    rel = np.abs(z_bp[0] - z_df[0]).max() / np.abs(z_bp[0]).max()
    assert rel < 5e-2, rel                              # 两口径 Z 一致（均无偏）
    # bypass 后的 Z 仍与理论一致（地板由 CTS 阶跃×连续谱决定，<1e-1）
    from synthts.model import load_site
    th = dict(mtprocess.theoretical_impedance(
        os.path.join(REPO, 'examples', 'mta.XMLSITE')))[f]
    err = np.abs(z_bp[0] - th).max() / np.abs(th).max()
    assert err < 1e-1, err


def test_gate7_amplitude_and_burst_stats(tmp_path):
    """闸门⑦：带内包络幅度分布与突发段长（两样本 KS vs 独立合成参照）。

    参照 = 同一估计器跑在同参数独立 seed 的合成上（设计⑦(b) 口径——
    估计器固有偏置在参照中原样呈现，检验的是统计可复现性）。
    """
    import calibrate_real as cal
    f_b = 1.0

    def env_ln_and_runs(seed):
        out = str(tmp_path / ('g7_%d' % seed))
        _run_noise_inproc(out, 3600, seed)
        (hx, hy), _ = cal.band_envelope(out, 'MTA', f_b)
        shutil.rmtree(out, ignore_errors=True)
        return (np.log(np.maximum(hx, 1e-300)),
                np.asarray(cal.burst_lengths(hx), dtype=float),
                float(np.corrcoef(hx, hy)[0, 1]))

    test_ln, test_runs, pol = env_ln_and_runs(4242)
    ref_pool_ln, ref_pool_runs = [], []
    for seed in (1, 2):
        a, b, _ = env_ln_and_runs(seed)
        ref_pool_ln.append(a)
        ref_pool_runs.append(b)
    ref_ln = np.concatenate(ref_pool_ln)
    ref_runs = np.concatenate(ref_pool_runs)

    def ks2(a, b):
        allv = np.concatenate([a, b])
        cdf1 = np.searchsorted(np.sort(a), allv, side='right') / len(a)
        cdf2 = np.searchsorted(np.sort(b), allv, side='right') / len(b)
        n, m = len(a), len(b)
        d = float(np.abs(cdf1 - cdf2).max())
        return d, 1.63 * math.sqrt((n + m) / (n * m))   # α=1% 渐近临界

    d, crit = ks2(test_ln, ref_ln)
    assert d < crit, (d, crit)                          # ⑦(a) 幅度分布
    if len(test_runs) >= 20 and len(ref_runs) >= 40:
        d2, crit2 = ks2(test_runs, ref_pool_runs[0])    # ⑦(b) 段长（弱化：单参照池首组）
        assert d2 < crit2 * 1.3, (d2, crit2)
    assert abs(pol) < 0.3, pol                          # ⑦(c) 极化独立


@pytest.mark.slow
def test_calibration_selftest(tmp_path):
    """§5.4 降级路径：合成自标定往返（tools/calibrate_real.py --selftest）。"""
    import subprocess as sp
    r = sp.run([sys.executable, 'tools/calibrate_real.py', '--selftest'],
               cwd=REPO, capture_output=True, text=True,
               encoding='utf-8', errors='replace', timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'PASS' in r.stdout
