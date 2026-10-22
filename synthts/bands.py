# -*- coding: utf-8 -*-
"""统一带网格与滤波器组（设计文档 §2.1/§9.2 公共 API）。

带中心 f_b = f0·2^(k·s)（s=带间距倍频程数，默认 ½ oct，f0=1Hz，全档共享）。
带掩模 A_b(f) 为 log2 f 域的幅度升余弦：|u|≤s/4 平坦=1，s/4<|u|<3s/4 与
相邻带幅度互补（cos/sin），满足功率分割恒等式 Σ_b|A_b(f)|²=1（带内）。
v2 合成（L2 分带）与研究内容(1)的逐频段提取/扣除共用本模块，保证合成与
分离在同一频段基上操作（扣除的"近似精确"误差界见设计 §2.5/§9.2）。
"""
import math

import numpy as np

BAND_REF_FREQ = 1.0      # 带网格基准（设计 §2.2，全档共享）
BAND_OCTAVE = 0.5        # 带间距（倍频程）


def band_centers(f_lo, f_hi, spacing=BAND_OCTAVE, ref=BAND_REF_FREQ):
    """[f_lo, f_hi] 内的带中心列表（升序）。

    端点截断口径：带中心落入 (f_lo/2^(s/2), f_hi·2^(s/2)] 即认为覆盖该端
    ——边缘带在截断侧无互补邻居，功率互补仅在完整网格上成立。
    """
    if f_lo <= 0 or f_hi <= f_lo:
        return []
    k_lo = math.ceil(math.log2(f_lo / ref) / spacing - 0.5)
    k_hi = math.floor(math.log2(f_hi / ref) / spacing + 0.5)
    return [ref * 2.0 ** (k * spacing) for k in range(k_lo, k_hi + 1)]


def band_amplitude(freqs, f_b, spacing=BAND_OCTAVE):
    """带 b 的幅度掩模 A_b(f)（平坦 1 / 升余弦过渡 / 0）。"""
    f = np.asarray(freqs, dtype=float)
    u = np.abs(np.log2(f / f_b))
    flat, edge = 0.25 * spacing, 0.75 * spacing
    a = np.zeros_like(f)
    m = u <= flat
    a[m] = 1.0
    m = (u > flat) & (u < edge)
    a[m] = np.cos(0.5 * math.pi * (u[m] - flat) / (edge - flat))
    return a


def power_complementarity_error(freqs, spacing=BAND_OCTAVE):
    """Σ_b|A_b|² 与 1 的最大偏差（网格完备处的自检；边缘截断区除外）。"""
    f = np.asarray(freqs, dtype=float)
    centers = band_centers(f.min(), f.max(), spacing)
    total = np.zeros_like(f)
    for f_b in centers:
        total += band_amplitude(f, f_b, spacing) ** 2
    inner = (f >= centers[0] * 2 ** (0.25 * spacing)) & \
            (f <= centers[-1] * 2 ** (-0.25 * spacing))
    return float(np.abs(total[inner] - 1.0).max()) if inner.any() else 0.0
