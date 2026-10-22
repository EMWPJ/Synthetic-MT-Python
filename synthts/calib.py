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
import os

from .cts_amt import CTS_AMT
from .cts_mt import CTS_MT

ADC_FACTOR = 1.0 / 8388608.0

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


def compute_responds(freqs):
    """按频率求系统响应（band 组内最近邻 + adcFactor 变换）。

    返回 responds[5][N]；无法覆盖的频率（band 组缺失）按单位 1E-6/1E-9 常数
    （主工程 NoCal 语义兜底）。
    """
    out = [[] for _ in range(5)]
    for f in freqs:
        band = band_of_freq(f)
        z5 = None
        if band in _MT:
            z5 = _nearest(_MT[band], f)
        elif band in _AMT:
            z5 = _nearest(_AMT[band], f)
        if z5 is None:
            for c in range(5):
                out[c].append(complex(1e-6, 0) if c < 2 else complex(1e-9, 0))
            continue
        for c in range(5):
            out[c].append(transform_cts(z5[c]))
    return out
