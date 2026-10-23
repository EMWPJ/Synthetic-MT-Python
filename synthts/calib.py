# -*- coding: utf-8 -*-
"""内置标定（纯数据，无外部引擎）。

两张内置 CTS 表（同一默认仪器：box 4360 + 线圈 BMT50000/53014/53025，
资产见 assets/）：
  CTS_MT  — MT 处理网格 50 频点（band 3/4/5，SysCal_V7.exe 金标输出固化）
  CTS_AMT — AMT 处理网格 40 频点（band 2-5，syscal_dll 于同网格预计算）

取值方式：按请求频率派生 band → 在对应 band 组内取最近网格点（不做插值——
该盒 CLB 真值在网格间符号交替，插值会相消失真；模型频点取处理网格频率即精确
命中）。band2 仅 AMT 表覆盖；band3-5 优先 MT 表（MT 合成主用），缺失回落 AMT 表。

输出合同（与主工程一致）：responds[ch] = adcFactor * conj(z) / |z|^2，
adcFactor = 1/2^23。
"""
import math
import os

import numpy as np

from .cts_amt import CTS_AMT
from .cts_mt import CTS_MT

ADC_FACTOR = 1.0 / 8388608.0

# 响应取值模式：'nearest'（默认，与主工程 InterpolateCTSResponses 一致）
# 或 'smooth'（符号解卷绕 + log-f 域复数线性插值——稠密 CTS 实验证明底层
# 响应平滑、表内"符号交替"仅为 z→−z 约定翻转；渲染与处理两侧须同用
# 同一模式， staircase 的核内混合偏置即消失）。
SMOOTH_RESPONDS = False

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
    """band 组的稠密响应（cts_dense，已变换、已对齐、log10 f 升序）。"""
    if band not in _SMOOTH_CACHE:
        try:
            from .cts_dense import DENSE_CTS
            rows = sorted((r for r in DENSE_CTS if r[1] == band), key=lambda x: x[0])
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
