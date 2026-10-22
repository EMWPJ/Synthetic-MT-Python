# -*- coding: utf-8 -*-
"""v2 噪声源引擎：按记录带合成渲染（设计文档 §3，方案 (a′)）。

架构合同：五通道均为同一对源过程 {P1, P2} 的线性组合
    X_c(t) = Σ_p T_{c,p}(f) · P_p(t)，T = C_{c,p}(f)（log 域插值场指纹）
    × responds_c(f)（CTS 最近邻）× μ0（磁道 A/m→B 折算，同 add_calibration）
禁止 Z=EH⁻¹ 推导路线（§1 原则一）。逐 bin 线性代数 ⇒ E(f_k)=Z(f_k)H(f_k)
精确（§2.4）。窗口档每记录独立渲染；连续档整轴单记录；记录等长 ⇒ 各档
bin 网格与 T 每档装配一次、全部记录复用（§3.2）。
"""
import math
from dataclasses import dataclass, field as dc_field

import numpy as np

from . import bands as bands_mod
from . import calib

MU0 = 4e-7 * math.pi


@dataclass
class RecordSpec:
    sample_rate: float
    n_samples: int
    begin: float                 # TDateTime 天数


@dataclass
class RecordPlan:
    """渲染记录计划（设计 §9.4 冻结类型；内置 TBL 为其一种构造器，
    M4 增"从实测 TS 提取"构造器——变长记录走逐记录 T 装配降级路径）。"""
    name: str
    continuous: bool
    records: list = dc_field(default_factory=list)


def plan_from_schemas(schemas, btime, etime):
    """内置 TBL 记划构造器：包装 synth.schema_segments（等长记录）。"""
    from .synth import schema_segments
    plans = {}
    for schema in schemas:
        plan = RecordPlan(schema['name'], schema['continuous'])
        for begin, n in schema_segments(schema, btime, etime):
            plan.records.append(RecordSpec(schema['sample_rate'], n, begin))
        plans[plan.name] = plan
    return plans


def interp_log_complex(anchor_freqs, values, freq_bins):
    """场指纹 log10(f) 域复线性插值到 bin 网格（唯一插值实现，§3.2 合同）。

    与 model._interp1 同语义（re/im 分别、端点钳制）；锚点处严格等于原值。
    """
    order = sorted(range(len(anchor_freqs)), key=lambda i: anchor_freqs[i])
    lf = np.log10(np.asarray(anchor_freqs, dtype=float)[order])
    v = np.asarray(values, dtype=complex)[order]
    fb = np.asarray(freq_bins, dtype=float)
    with np.errstate(divide='ignore'):      # bin 0（DC）→ -inf，np.interp 钳到首锚点
        lf_b = np.log10(fb)
    re = np.interp(lf_b, lf, v.real)
    im = np.interp(lf_b, lf, v.imag)
    return re + 1j * im


def transfer_matrix(site, freq_bins, scale=1.0):
    """装配 T[c, p]（complex128，(5, 2, n_bins)）。

    site 为已 nature 归一、未标定的原始场（noise 路径不走 add_calibration：
    标定以逐 bin 最近邻 responds 施加，保持与 mtprocess 处理端一致）。
    """
    fehs = site.get_feh(noise=False)
    anchors, cols = fehs[0], fehs[1:]           # cols: ex1..hz1, ex2..hz2
    resp = calib.responds_on_grid(freq_bins)    # [5][n_bins]
    bins = np.asarray(freq_bins, dtype=float)
    T = np.empty((5, 2, len(bins)), dtype=np.complex128)
    for c in range(5):
        if c < 2:                                # E 道：E_adc = E / rE
            gain = 1.0 / resp[c]
        else:                                    # H 道：H_adc = H·μ0 / rH
            gain = MU0 / resp[c]
        for p in range(2):
            T[c, p] = interp_log_complex(anchors, cols[p * 5 + c], bins) * gain
    return T * scale


def render_record_source(rng, n, sample_rate, alpha1=1.0):
    """L1 源：单位谱密度（每 Hz）宽带白噪声 → 每样本方差 = fs。

    每样本方差取 fs（而非 1）使各采样档渲染出的物理谱密度一致（TS3∩TS4
    等重叠频段的源相同）——"单位 PSD"按每 Hz 定义。返回两极化 rfft 谱
    （DC 置零，§3.2）。
    """
    amp = math.sqrt(alpha1 * sample_rate)
    P = []
    for _ in range(2):
        P.append(np.fft.rfft(rng.standard_normal(n)) * amp)
    P[0][0] = 0.0
    P[1][0] = 0.0
    return P


def render_site(P, T):
    """逐站：X_c = IFFT(Σ_p T[c,p]·P̂_p) → (5, n) float32。"""
    nb = len(P[0])
    n = 2 * (nb - 1) if nb % 2 == 1 else 2 * nb - 1   # rfft 输出长↔原始 n
    X = np.empty((5, n), dtype=np.float32)
    for c in range(5):
        spec = T[c, 0] * P[0] + T[c, 1] * P[1]
        X[c] = np.fft.irfft(spec, n=n).astype(np.float32)
    return X


def synthesize_noise(site_list, schema, plan, rng, alpha1=1.0):
    """渲染一批测点的一个 TS 档（源跨测点逐 bin 共享，§3.1）。

    返回与 synth.synthetic_sites 同构的 ts dict 列表（write_time_series
    直接消费）。记录等长假设：T 每档装配一次（§3.5 协议注释）。
    """
    if not plan.records:
        return [{'name': schema['name'], 'sample_rate': schema['sample_rate'],
                 'begins': [], 'continuous': schema['continuous'],
                 'channels': [[] for _ in range(5)]} for _ in site_list]
    n0 = plan.records[0].n_samples
    assert all(r.n_samples == n0 for r in plan.records), \
        'RecordPlan 等长假设不成立（变长记录走 M4 逐记录装配路径）'
    freqs = np.fft.rfftfreq(n0, d=1.0 / plan.records[0].sample_rate)
    Ts = [transfer_matrix(s, freqs) for s in site_list]
    outs = [np.empty((5, n0 * len(plan.records)), dtype=np.float32)
            for _ in site_list]
    offs = np.arange(len(plan.records) + 1) * n0
    sr0 = plan.records[0].sample_rate
    for j, _rec in enumerate(plan.records):
        P = render_record_source(rng, n0, sr0, alpha1)
        for k in range(len(site_list)):
            X = render_site(P, Ts[k])
            outs[k][:, offs[j]:offs[j + 1]] = X
    result = []
    for k, site in enumerate(site_list):
        channels = [[outs[k][c][offs[j]:offs[j + 1]]
                     for j in range(len(plan.records))] for c in range(5)]
        result.append({'name': schema['name'],
                       'sample_rate': plan.records[0].sample_rate,
                       'begins': [r.begin for r in plan.records],
                       'continuous': plan.continuous,
                       'channels': channels})
    return result


def band_count_for_plan(plan, model_f_lo, model_f_hi):
    """档内有效带数（摘要/闸门用；下截 max(模型下限, 2/总时长)、上截 0.4·sr）。"""
    if not plan.records:
        return 0
    sr = plan.records[0].sample_rate
    total_s = sum(r.n_samples for r in plan.records) / sr
    f_lo = max(model_f_lo, 2.0 / total_s)
    f_hi = min(model_f_hi, 0.4 * sr)
    return len(bands_mod.band_centers(f_lo, f_hi)) if f_hi > f_lo else 0
