# -*- coding: utf-8 -*-
"""作者最小数据配方实测：数据 ≥8 周期、每窗 4 周期、两窗 LSQ。

配方：窗长 = 4 周期（Hann、精确频点核），数据段 = N 个相邻不重叠窗，
N=2 即作者配方（数据 8 周期）。N ∈ {2,4,8,16} 扫数据量。
起点网格化扫过全部记录段 → 每频点误差分布（median/p90/max）。

引擎对照：
  noise = 当前 v2.2 连续谱引擎（syn24h，24h natural 记录）
  lines = v1 稀疏梳齿引擎（demo_v22_lines/sparse，作者当年配方的语境）

频点→档：TS5 [0.1,0.32,1,3.2]；TS4 [10,32]；TS3 [100,320]。
物理分段由记录头日历时钟重建（相邻头时差 = 扫描时长为连续，时间跳变 =
段间空隙）；窗口不跨段。
理论 = 模型锚点场 E·H⁻¹（锚点处与任何插值口径严格一致）。
指标 = Zxy 的 ρa 相对误差与相位误差（max-分量口径会被小量级 Zyy 放大，
仅作参考列）。
"""
import io
import json
import math
import os
import sys

import numpy as np

from synthts import calib
from synthts.mtprocess import theoretical_impedance

MU0 = 4e-7 * math.pi

CASES = {
    'noise': {'dir': 'syn24h', 'model': 'syn24h/MTA_dense.XMLSITE'},
}
BAND_RANGE = {'TS5': (0.0, 4.0), 'TS4': (4.0, 75.0), 'TS3': (75.0, 1e9)}
NS = [2, 4, 8, 16]
WPS = [4.0, 8.0]          # 窗长（周期数）：4=作者配方，8=当前缺省
MAX_BLOCKS = 240


def _tsec(head):
    """32B 头的日历时钟 → era 秒。"""
    sec, minute, hour = head[0], head[1], head[2]
    day, month, year = head[3], head[4], head[5]
    y = year + (2000 if head[7] == 20 else 1900)
    days = (np.datetime64('%04d-%02d-%02d' % (y, month, day), 's')
            - np.datetime64('1970-01-01', 's')) / np.timedelta64(1, 's')
    return int(days) + hour * 3600 + minute * 60 + sec


def read_ts_segments(path):
    """读 TS 文件 → (sr, data[5][n], segs=[(start, len)])。

    段 = 头时钟连续（时差恰为扫描时长）的极大记录串；时间跳变即段间空隙。
    """
    raw = open(path, 'rb').read()
    chans = [[] for _ in range(5)]
    recs = []
    sr = None
    off = 0
    total = 0
    while off + 32 <= len(raw):
        head = raw[off:off + 32]
        scans = int.from_bytes(head[10:12], 'little')
        ch = head[12]
        if not (0 < scans <= 100000 and 0 < ch <= 8):
            break
        nbytes = ch * scans * 3
        if len(raw) < off + 32 + nbytes:
            break
        sr = int.from_bytes(head[18:20], 'little')
        b = np.frombuffer(raw[off + 32:off + 32 + nbytes],
                          dtype=np.uint8).reshape(-1, 3)
        v = b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) \
            | (b[:, 2].astype(np.int32) << 16)
        v = np.where(v >= (1 << 23), v - (1 << 24), v)
        m = v.reshape(scans, ch).T.astype(np.float64)
        for c in range(min(ch, 5)):
            chans[c].append(m[c])
        recs.append((_tsec(head), scans, total))
        total += scans
        off += 32 + nbytes
    data = [np.concatenate(chans[c]) if chans[c] else np.zeros(0)
            for c in range(5)]
    segs = []
    run_start, run_end_t, run_len = None, None, 0
    for t, scans, start in recs:
        if run_start is None:
            run_start, run_end_t, run_len = start, t, scans
        elif t - run_end_t == 1:                # 相邻记录头差 1 秒 = 连续
            run_len += scans
            run_end_t = t
        else:
            segs.append((run_start, run_len))
            run_start, run_end_t, run_len = start, t, scans
    if run_start is not None:
        segs.append((run_start, run_len))
    return sr, data, segs


def block_starts(segs, n, N):
    """每段内 N 个相邻窗的块起点（窗不跨段）。"""
    out = []
    for g, L in segs:
        s = g
        while s + N * n <= g + L:
            out.append(s)
            s += n
    return out


def win_dfts(data, starts, n, N, f, fs):
    """给定块起点：N 相邻窗 × 4 通道的精确频点加窗 DFT。"""
    kern = np.hanning(n) * np.exp(-2j * math.pi * f * np.arange(n) / fs)
    offs = (np.asarray(starts)[:, None, None]
            + np.arange(N)[None, :, None] * n
            + np.arange(n)[None, None, :])                       # (B, N, n)
    out = np.empty((4, len(starts), N), dtype=complex)
    for c in range(4):
        out[c] = (data[c][offs] * kern[None, None, :]).sum(axis=2)
    return out                                                   # (4, B, N)


def run_case(tag, cfg):
    d = cfg['dir']
    theory = dict(theoretical_impedance(cfg['model']))
    cache = {}
    results = {}
    for band, (blo, bhi) in BAND_RANGE.items():
        cands = sorted(k for k in theory if blo <= k <= bhi)
        if not cands:
            continue
        if len(cands) > 4:                     # 每档最多 4 个频点（均匀抽样）
            idx = np.unique(np.linspace(0, len(cands) - 1, 4).round().astype(int))
            cands = [cands[i] for i in idx]
        for f in cands:
            path = os.path.join(d, 'MTA.%s' % band)
            if band not in cache:
                cache[band] = read_ts_segments(path)
            sr, data, segs = cache[band]
            zt = theory[f]
            r = calib.compute_responds([f])
            for wp in WPS:
                n = int(round(wp * sr / f))
                if n < 12 or sr / f < 4.0:
                    continue
                for N in NS:
                    starts = block_starts(segs, n, N)
                    if not starts:
                        continue
                    if len(starts) > MAX_BLOCKS:
                        starts = starts[::int(math.ceil(len(starts) / MAX_BLOCKS))][:MAX_BLOCKS]
                    V = win_dfts(data, np.asarray(starts), n, N, f, sr)
                    hx = V[2] * r[2][0] / MU0                        # (B,N)
                    hy = V[3] * r[3][0] / MU0
                    ex = V[0] * r[0][0]
                    ey = V[1] * r[1][0]
                    errs, rho_err, ph_err, rhoa = [], [], [], []
                    for b in range(len(starts)):
                        h = np.column_stack([hx[b], hy[b]])
                        B = np.column_stack([ex[b], ey[b]])
                        z = np.linalg.lstsq(h, B, rcond=None)[0].T
                        errs.append((np.abs(z - zt) / np.abs(zt)).max())
                        rhoa.append(abs(z[0, 1]) ** 2 / (MU0 * 2 * math.pi * f))
                        rho_err.append(abs(abs(z[0, 1]) / abs(zt[0, 1]) - 1))
                        ph_err.append(math.degrees(math.atan2(
                            z[0, 1].imag, z[0, 1].real) - math.atan2(
                            zt[0, 1].imag, zt[0, 1].real)))
                    results[(f, wp, N)] = dict(
                        errs=np.array(errs), rho_err=np.array(rho_err),
                        ph_err=np.array(ph_err), rhoa=np.array(rhoa),
                        nblk=len(starts))
    return results


def main():
    all_res = {}
    for tag, cfg in CASES.items():
        if not os.path.isdir(cfg['dir']):
            print('skip %s (no data dir %s)' % (tag, cfg['dir']))
            continue
        all_res[tag] = run_case(tag, cfg)
        print('=== %s ===' % tag)
        print('%8s %4s %4s %6s %24s %12s %12s %7s' % (
            'f[Hz]', 'win', 'N', 'nblk', 'errComp med/p90',
            'rhoa med/p90', 'phi_med[deg]', 'blowup'))
        for (f, wp, N), res in sorted(all_res[tag].items()):
            e = res['errs']
            blow = float((e > 1.0).mean())
            print('%8g %4.0f %4d %6d %10.3e/%.3e %8.3e/%.3e %12.3e %7.1f%%' % (
                f, wp, N, res['nblk'], np.median(e), np.percentile(e, 90),
                np.median(res['rho_err']), np.percentile(res['rho_err'], 90),
                np.median(np.abs(res['ph_err'])), 100 * blow))
    import json
    out = {'%r|%g|%d' % k: {kk: [float(v) for v in vv] if isinstance(vv, np.ndarray)
                            else vv for kk, vv in d.items()}
           for tag_res in all_res.values() for k, d in tag_res.items()}
    with io.open('plot_out/min_recipe_results.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh)
    print('saved plot_out/min_recipe_results.json')


if __name__ == '__main__':
    sys.exit(main())
