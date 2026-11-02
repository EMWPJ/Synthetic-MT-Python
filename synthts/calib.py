# -*- coding: utf-8 -*-
"""内置标定（纯数据，无外部引擎）。

两张内置 CTS 表（同一默认仪器：box 4360 + 线圈 BMT50000/53014/53025，
资产见 assets/）：
  CTS_MT  — MT 处理网格 50 频点（band 3/4/5，SysCal_V7.exe 金标输出固化）
  CTS_AMT — AMT 处理网格 40 频点（band 2-5，syscal_dll 于同网格预计算）

取值方式（2026-11-02 起默认 smooth）：符号解卷绕 + log-f 域复数线性插值于
稠密 CTS 表（cts_dense，syscal_dll 于稠密网格预计算）——与 MTDataPro 原生
SysCal 逆向库的平滑连续求值对齐：合成端烤进 counts 的响应与处理端除去的
响应在任意方案网格上逐 bin 一致，阶梯残差消失（nearest 模式在非锚点方案
网格上曾致低频 ρa ±10/15% 锯齿，见 AGENTS 陷阱清单）。band2 仅 AMT 表
覆盖；band3-5 优先 MT 表（MT 合成主用），缺失回落 AMT 表。
`SMOOTH_RESPONDS = False` 可回退 legacy nearest 模式（与主工程
InterpolateCTSResponses 一致；仅用于复现旧行为，渲染/处理两侧须同切）。

输出合同（与主工程一致）：responds[ch] = adcFactor * conj(z) / |z|^2，
adcFactor = 1/2^23。
"""
import math
import os

import numpy as np

from .cts_amt import CTS_AMT
from .cts_mt import CTS_MT

ADC_FACTOR = 1.0 / 8388608.0

# 响应取值模式：True（默认）= 稠密 CTS 解卷绕平滑插值，对齐 MTDataPro
# 原生 SysCal 平滑去标定；False = legacy 50 点网格最近邻（仅复现旧行为）。
# 渲染与处理两侧须同用同一模式。
SMOOTH_RESPONDS = True

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')


def band_of_freq(f: float) -> int:
    """SyntheticBandOfFreq: >400→2, >8→3, >0.8→4, else 5。"""
    if f > 400:
        return 2
    if f > 8:
        return 3
    if f > 0.8:
        return 4
    return 5


def _parse(table):
    groups = {}
    for f, b, vals in table:
        z5 = [complex(vals[c * 2], vals[c * 2 + 1]) for c in range(5)]
        groups.setdefault(b, []).append((f, z5))
    for b in groups:
        groups[b].sort(key=lambda x: x[0])
    return groups


_MT = _parse(CTS_MT)
_AMT = _parse(CTS_AMT)


def transform_cts(z: complex) -> complex:
    """z(CTS 语义) → adcFactor * conj(z)/|z|^2（InterpolateCTSResponses 输出合同）。"""
    denom = z.real * z.real + z.imag * z.imag
    if denom < 1e-60:
        denom = 1e-60
    return complex(ADC_FACTOR * z.real / denom, -ADC_FACTOR * z.imag / denom)


def _nearest(group, f):
    best, bestd = None, None
    for gf, z5 in group:
        d = abs(gf - f)
        if bestd is None or d < bestd:
            best, bestd = z5, d
    return best


def _unwrap_group(rows):
    """组内相位解卷绕：逐行翻转让相邻行相关为正（z→−z 约定翻转还原）。"""
    out = [rows[0]]
    for z in rows[1:]:
        prev = out[-1]
        zc = z if (z.real * prev.real + z.imag * prev.imag) >= 0 else -z
        out.append(zc)
    return out


_SMOOTH_CACHE = {}


def _dense_group(band):
    """band 组的稠密响应（cts_dense，已变换、已对齐、log10 f 升序）。

    两级行过滤（数据文件保持 dll 原始输出不动，过滤在载入时做）：
    ① 任一通道恰为 0 的行是 dll 出界点（如 Hx ~12kHz），丢弃——否则网格
       边缘外 bin 的插值钳位到零行会得零响应；
    ② 局部离群行（任一通道模长偏离 ±2 邻行中位数 20 倍以上）是陷波中心/
       边缘的退化求值（band3 46.2–50.05 Hz 七行、band5 0.3866 Hz 一行，
       邻比 4e-23…5e+22）——log-f 线性插值的地雷，丢弃后跨缺口线性内插。
       判据 20 倍安全：稠密网格步距 ~0.3%，真响应物理上不可能逐行跳 20 倍；
       陷波的陡峭但平滑的行（邻比 0.4–2.5）保留。
    """
    if band not in _SMOOTH_CACHE:
        try:
            from .cts_dense import DENSE_CTS
            rows = sorted((r for r in DENSE_CTS
                           if r[1] == band and all(v != 0 for v in r[2])),
                          key=lambda x: x[0])
            # ② 局部离群过滤（逐通道；两轮收敛——删行改变邻域）
            n = len(rows)
            if n >= 5:
                mags = [[abs(v) for v in r[2][0::2]] for r in rows]  # 5 通道模长
                drop = set()
                for i in range(n):
                    lo, hi = max(0, i - 2), min(n, i + 3)
                    for c in range(5):
                        nb = [mags[j][c] for j in range(lo, hi) if j != i]
                        med = sorted(nb)[len(nb) // 2]
                        if med > 0 and (mags[i][c] > 20.0 * med
                                        or mags[i][c] < 0.05 * med):
                            drop.add(i)
                            break
                if drop:
                    rows = [r for i, r in enumerate(rows) if i not in drop]
        except Exception:
            rows = []
        if not rows:
            _SMOOTH_CACHE[band] = None
            return None
        gf = np.array([r[0] for r in rows])
        z5 = [np.array([complex(*r[2][2 * c:2 * c + 2]) for r in rows])
              for c in range(5)]
        _SMOOTH_CACHE[band] = (gf, z5)
    return _SMOOTH_CACHE[band]


def _respond_at_smooth(band, f):
    g = _dense_group(band)
    if g is None:
        return None
    gf, vals = g
    lg = math.log10(f)
    lgv = np.log10(gf)
    return [complex(float(np.interp(lg, lgv, v.real)),
                    float(np.interp(lg, lgv, v.imag))) for v in vals]


def compute_responds(freqs):
    """按频率求系统响应（band 组内最近邻 + adcFactor 变换）。

    SMOOTH_RESPONDS=True 时改用解卷绕平滑插值（渲染/处理两侧须同用）。
    返回 responds[5][N]；无法覆盖的频率（band 组缺失）按单位 1E-6/1E-9 常数
    （主工程 NoCal 语义兜底）。
    """
    out = [[] for _ in range(5)]
    for f in freqs:
        band = band_of_freq(f)
        if SMOOTH_RESPONDS:
            z5 = _respond_at_smooth(band, f)   # 已变换
        else:
            z5 = None
            if band in _MT:
                z5 = _nearest(_MT[band], f)
            elif band in _AMT:
                z5 = _nearest(_AMT[band], f)
        if z5 is None:
            for c in range(5):
                out[c].append(complex(1e-6, 0) if c < 2 else complex(1e-9, 0))
            continue
        if not SMOOTH_RESPONDS:
            z5 = [transform_cts(v) for v in z5]
        for c in range(5):
            out[c].append(z5[c])
    return out


# ── 预变换的网格缓存（responds_on_grid 用；_parse 后各组已按频率升序）──────
_GRID_CACHE = {}


def _grid_for(band):
    if band not in _GRID_CACHE:
        group = _MT.get(band) or _AMT.get(band)
        if group is None:
            _GRID_CACHE[band] = None
        else:
            import numpy as np
            gf = np.array([gf for gf, _ in group])
            tv = np.array([[transform_cts(z5[c]) for _, z5 in group]
                           for c in range(5)])
            _GRID_CACHE[band] = (gf, tv)
    return _GRID_CACHE[band]


def responds_on_grid(freqs):
    """compute_responds 的向量化入口（逐点语义恒等，供 v2 逐 bin 装配）。

    SMOOTH_RESPONDS=True 时改用解卷绕平滑插值。
    返回 [5] 个 numpy complex 数组，长度同 freqs。最近邻平局时保留较左侧
    网格点（与 _nearest 的"严格更优才替换"语义一致）。
    """
    import numpy as np
    f = np.asarray(freqs, dtype=float)
    if SMOOTH_RESPONDS:
        bands = np.where(f > 400, 2, np.where(f > 8, 3, np.where(f > 0.8, 4, 5)))
        out = [np.empty(len(f), dtype=complex) for _ in range(5)]
        for band in (2, 3, 4, 5):
            idx = np.nonzero(bands == band)[0]
            if not len(idx):
                continue
            g = _dense_group(band)
            if g is None:
                continue
            gf, vals = g
            lgb = np.log10(np.maximum(f[idx], 1e-300))
            lgv = np.log10(gf)
            for c in range(5):
                out[c][idx] = (np.interp(lgb, lgv, vals[c].real)
                               + 1j * np.interp(lgb, lgv, vals[c].imag))
        return out
    bands = np.where(f > 400, 2, np.where(f > 8, 3, np.where(f > 0.8, 4, 5)))
    out = [np.empty(len(f), dtype=complex) for _ in range(5)]
    for band in (2, 3, 4, 5):
        idx = np.nonzero(bands == band)[0]
        if not len(idx):
            continue
        grid = _grid_for(band)
        if grid is None:
            for c in range(5):
                out[c][idx] = 1e-6 if c < 2 else 1e-9
            continue
        gf, tv = grid
        pos = np.clip(np.searchsorted(gf, f[idx]), 1, len(gf) - 1)
        d_lo = np.abs(gf[pos - 1] - f[idx])
        d_hi = np.abs(gf[pos] - f[idx])
        j = np.where(d_hi < d_lo, pos, pos - 1)
        for c in range(5):
            out[c][idx] = tv[c][j]
    return out
