# -*- coding: utf-8 -*-
"""实测/合成 TS 的源参数统计标定（设计 §5.4；M2 交付）。

从连续档时间序列提取指定带内的滑窗 DFT 幅度包络，估计段增益 LN σ
（含 Rayleigh 贡献 ≈ sqrt(σ_g² + π²/24)）、突发段长中位数与 LN σ、
两道包络相关（极化独立性）。结果打印为 JSON（不写文件——调用方自行
重定向），可直接用于 `-source-params` 覆盖项。

用法（在仓库根目录执行）：
  python tools/calibrate_real.py <TS目录> <测点名> <带中心Hz>
合成自标定（D7 降级路径）：
  python tools/calibrate_real.py --selftest   # 合成 natural 已知参数 → 估计往返
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthts import mtprocess                          # noqa: E402

WINDOW_PERIODS = 8.0


def band_envelope(ts_dir, site, f_b):
    """带通（±0.3 oct 升余弦）+ Hilbert 包络，按窗长步长抽样。

    单频 DFT 估计器的裙摆会吃进 2~3 个相邻带的独立增益（σ_g² 被稀释
    ~1/2.8），带通+Hilbert 才能隔离单带（研究内容① 的原型口径）。
    """
    picked = mtprocess.pick_ts_file(f_b, ts_dir, site)
    if picked is None:
        raise SystemExit('no TS band covers %.4g Hz' % f_b)
    sr, data = picked[1], picked[2]
    n = int(round(WINDOW_PERIODS * sr / f_b))
    if n < 8:
        raise SystemExit('window too short at %.4g Hz' % f_b)
    freqs = np.fft.rfftfreq(len(data[0]), 1.0 / sr)
    with np.errstate(divide='ignore'):
        lf = np.log2(np.maximum(freqs, 1e-300) / f_b)
    mask = np.zeros_like(freqs)
    m = np.abs(lf) <= 0.3
    mask[m] = 1.0
    tr = (np.abs(lf) > 0.3) & (np.abs(lf) < 0.45)
    mask[tr] = 0.5 - 0.5 * np.cos(math.pi * (0.45 - np.abs(lf[tr])) / 0.15)
    out = []
    for c in (2, 3):                                   # Hx, Hy
        X = np.fft.rfft(data[c]) * mask
        half = np.zeros(len(X) + 1, dtype=complex)     # 单边谱 → 解析信号
        half[1:len(X)] = 2.0 * X[1:]
        a = np.fft.ifft(half)[:len(data[c])]
        out.append(np.abs(a)[::n])
    return out, sr


def _smooth(x, k):
    return np.convolve(x, np.ones(k) / k, mode='same')


def ln_sigma(env):
    return float(np.std(np.log(np.maximum(env, 1e-300))))


def burst_lengths(env):
    """4 点平滑包络过中位阈值的连通段长（窗数计；平滑宽 ≪ 段长）。"""
    on = _smooth(env, 4) > np.median(_smooth(env, 4))
    lens, cur = [], 0
    for v in on:
        if v:
            cur += 1
        elif cur:
            lens.append(cur)
            cur = 0
    if cur:
        lens.append(cur)
    return lens


def calibrate(ts_dir, site, f_b):
    (hx, hy), fs = band_envelope(ts_dir, site, f_b)
    params = {
        'band_hz': f_b,
        'sigma_env_ln': ln_sigma(hx),
        'sigma_env_ln_hy': ln_sigma(hy),
        'pol_corr': float(np.corrcoef(hx, hy)[0, 1]),
        'n_windows': int(len(hx)),
    }
    lens = burst_lengths(hx)
    if lens:
        params['burst_median_windows'] = float(np.median(lens))
        params['burst_median_s'] = float(np.median(lens)) * WINDOW_PERIODS / f_b
        params['burst_len_sigma_ln'] = float(
            np.std(np.log(np.maximum(lens, 1))) if len(lens) > 2 else 0.0)
    return params


def _selftest():
    """合成自标定往返：natural 已知 σ_g=6dB、T̄=64 周期 → 估计核对。"""
    from synthts.cli import main as synth_main
    import contextlib
    import io
    import shutil
    out = 'calib_tmp'
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    with contextlib.redirect_stdout(io.StringIO()):
        rc = synth_main(['-model', 'examples/mta.XMLSITE', '-out', out,
                         '-dur', '14400', '-seed', '42', '-scale', '10',
                         '-dense', '0', '-source', 'noise'])
    assert rc == 0
    f_b = 1.0
    p = calibrate(out, 'MTA', f_b)
    shutil.rmtree(out, ignore_errors=True)
    # 期望：σ_env² ≈ σ_g² + π²/24（LN 增益 × Rayleigh，§2.3；残余 ~0.1 来自
    # 过渡区邻带尾稀释）。段长：过全局中位阈的连通段对 LN 段群有固有 ~减半
    # 偏置（短段无过阈段、长段部分过阈）——估计器偏置已知，往返核对用
    # factor-4 宽容差（设计⑦(b)：参照口径=同估计器，不主张绝对真值恢复）
    sigma_g_ln = 6.0 / 8.685889638065035
    exp_ln = math.sqrt(sigma_g_ln ** 2 + math.pi ** 2 / 24.0)
    ok_sigma = abs(p['sigma_env_ln'] - exp_ln) < 0.15
    exp_len_s = 64.0 / f_b
    ok_len = 0.25 < p['burst_median_s'] / exp_len_s < 1.5
    print('selftest: sigma_env=%.3f (期望 %.3f) %s | burst_med=%.1fs '
          '(期望 %.0fs) %s | pol_corr=%.2f'
          % (p['sigma_env_ln'], exp_ln, 'PASS' if ok_sigma else 'FAIL',
             p['burst_median_s'], exp_len_s, 'PASS' if ok_len else 'FAIL',
             p['pol_corr']))
    return 0 if (ok_sigma and ok_len and abs(p['pol_corr']) < 0.3) else 1


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == '--selftest':
        return _selftest()
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    params = calibrate(sys.argv[1], sys.argv[2], float(sys.argv[3]))
    import json
    print(json.dumps(params, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
