# -*- coding: utf-8 -*-
"""v2 三层源（M1: L1；M2: +L2 分带突发包络 + γ 公共因子 + L3 事件层）。

SourceProtocol 形态（设计 §3.5）：rng 分级派生（SeedSequence → 按名称排序
的 schema 子序列 → 记录噪声流 + 包络/事件流），源抽签全部在 schema 层与
测点无关——同源合同由结构保证。包络段链与事件定义在全局真实时间轴（跨记
录间隙连续）；带内白噪按记录独立抽签（间隙 ≫ 带相关时间时统计等价于连续
过程的离散采样，§3.3）。

rng 流说明：记录噪声（L1+L2 带白噪）与包络/事件分别从 schema 子序列的
两个派生 Generator 顺序消费——同参数同流逐位复现；带数/事件数等参数变化
会平移对应流的下游（记录内与包络内各自封闭，影响面见设计 §3.4 注）。

功率归一化：L2 权重 α2 预除 E[g²]·E[γ²]（解析值 exp(2σ_ln²)，§2.1），
使合同锚定均值 PSD；带集下截（带宽 <2.5 bin 的带跳过——该档低频端 L2
份额略低，谱形由跨档覆盖，处理按 pick_ts_file 选窗最多档，§2.5）。
"""
import math

import numpy as np

from . import bands as bands_mod
from . import render_fft

LN_OF_DB = 8.685889638065035      # 10/ln(10)：dB ↔ ln 换算
MIN_BURST_PERIODS = 10.0          # 段长下限（§2.5 谱展宽守卫）
CROSSFADE_PERIODS = 16.0          # 段间升余弦过渡宽（§2.1）


def _db_to_ln(sigma_db):
    return sigma_db / LN_OF_DB


def make_chain(rng, f_b, dur_s, mean_periods, sigma_len_ln, sigma_gain_ln):
    """包络段链：[(t0, t1, gain)] 覆盖 [0, dur_s]，段长/增益 log-normal。"""
    period = 1.0 / f_b
    segs = []
    t = 0.0
    while t < dur_s:
        ln_len = max(rng.normal(math.log(mean_periods), sigma_len_ln),
                     math.log(MIN_BURST_PERIODS))
        dt = math.exp(ln_len) * period
        segs.append((t, min(t + dt, dur_s),
                     math.exp(rng.normal(0.0, sigma_gain_ln))))
        t += dt
    return segs


def chain_arrays(segs):
    """段链 → (edges, gains) 数组（采样热路径预转换，避免每次重建）。"""
    return (np.array([s[0] for s in segs] + [segs[-1][1]]),
            np.array([s[2] for s in segs]))


def sample_chain(edges, gains, t_axis, fade_s):
    """段链在真实时间轴 t_axis（秒，单调升）上的采样：分段常数 + 升余弦。

    过渡按边界二分切片施加（切片总长 ≤ n + 边界区少量重叠）——整轴连续档
    的段数×轴长乘积不可接受（曾致 72h TS5 27min）。
    """
    out = np.empty(len(t_axis), dtype=np.float64)
    t_lo, t_hi = float(t_axis[0]), float(t_axis[-1])
    k_lo = int(np.clip(np.searchsorted(edges, t_lo - fade_s, side='right') - 1,
                       0, len(gains) - 1))
    k_hi = int(np.clip(np.searchsorted(edges, t_hi + fade_s, side='left'),
                       k_lo + 1, len(gains)))
    sub_edges, sub_gains = edges[k_lo:k_hi + 1], gains[k_lo:k_hi]
    idx = np.clip(np.searchsorted(sub_edges, t_axis, side='right') - 1,
                  0, len(sub_gains) - 1)
    out[:] = sub_gains[idx]
    if fade_s <= 0 or len(sub_gains) < 2:
        return out
    half = 0.5 * fade_s
    for k in range(1, len(sub_gains)):     # 每边界只碰自己的切片
        tb = sub_edges[k]
        i0 = int(np.searchsorted(t_axis, tb - half, side='left'))
        i1 = int(np.searchsorted(t_axis, tb + half, side='right'))
        if i1 <= i0:
            continue
        w = 0.5 - 0.5 * np.cos(
            math.pi * (t_axis[i0:i1] - (tb - half)) / fade_s)
        out[i0:i1] = sub_gains[k - 1] * (1.0 - w) + sub_gains[k] * w
    return out


def make_events(rng, dur_s, f_lo, f_hi, rate_per_hour, q, sigma_db):
    """L3 事件表：Poisson 到达 + log-uniform 中心频 + LN 幅度 + 随机极化。

    τ 与 Q 的关系按 §2.6 修正式 Q = 2√2π·f_c·τ_e（rms 时宽）。幅度标度
    约定（M2）：事件为源过程的时域加性成分，幅度中位数由调用方以
    sqrt(α3·fs) 折算（与单位 PSD 白噪同量纲、重尾向上）。
    """
    n = int(rng.poisson(rate_per_hour * dur_s / 3600.0)) if rate_per_hour > 0 else 0
    events = []
    for _ in range(n):
        t0 = float(rng.uniform(0.0, dur_s))
        fc = float(math.exp(rng.uniform(math.log(f_lo), math.log(f_hi))))
        tau = q / (2.0 * math.sqrt(2.0) * math.pi * fc)
        amp = math.exp(rng.normal(0.0, _db_to_ln(sigma_db)))
        events.append((t0, fc, tau, amp,
                       float(rng.uniform(0.0, 2.0 * math.pi)),   # phase
                       float(rng.uniform(0.0, 2.0 * math.pi))))  # pol angle
    return events


def _render_events(events, t_axis, a_ref):
    """事件在时间轴上的两极化贡献（源单位）。"""
    p1 = np.zeros(len(t_axis), dtype=np.float64)
    p2 = np.zeros(len(t_axis), dtype=np.float64)
    for t0, fc, tau, amp, phase, theta in events:
        dt = t_axis - t0
        m = np.abs(dt) <= 4.0 * tau
        if not m.any():
            continue
        wav = amp * np.exp(-0.5 * (dt[m] / tau) ** 2) \
            * np.cos(2.0 * math.pi * fc * dt[m] + phase)
        p1[m] += wav * math.cos(theta) * a_ref
        p2[m] += wav * math.sin(theta) * a_ref
    return p1, p2


class NoiseSource:
    """三层噪声源（L1 连续谱底 + L2 分带突发 + L3 事件，设计 §2.1）。"""

    def __init__(self, seed, params):
        self.seed = int(seed)
        self.alpha1 = float(params.get('alpha1', 1.0))
        self.alpha2 = float(params.get('alpha2', 0.0))
        self.alpha3 = float(params.get('alpha3', 0.0))
        if abs(self.alpha1 + self.alpha2 + self.alpha3 - 1.0) > 1e-9:
            raise SystemExit('alpha1+alpha2+alpha3 must be 1')
        self.band_octave = float(params.get('band_octave', 0.5))
        self.burst_periods = float(params.get('burst_periods', 64.0))
        self.burst_len_sigma = float(params.get('burst_len_sigma', 0.5))
        self.burst_sigma_db = float(params.get('burst_sigma_db', 6.0))
        self.gamma_sigma_db = float(params.get('common_sigma_db', 0.0))
        self.event_rate = float(params.get('event_rate', 0.0))
        self.event_q = float(params.get('event_q', 8.0))
        self.event_sigma_db = float(params.get('event_sigma_db', 10.0))
        # 功率归一（解析 E[g²]·E[γ²] = exp(2σ_g,ln²)·exp(2σ_γ,ln²)，§2.1）
        self._eg2 = math.exp(2.0 * _db_to_ln(self.burst_sigma_db) ** 2) \
            * math.exp(2.0 * _db_to_ln(self.gamma_sigma_db) ** 2)

    def schema_children(self, schema_names):
        """SeedSequence → 按名称排序派生子序列（§3.4 分级 rng）。"""
        ss = np.random.SeedSequence(self.seed)
        names = sorted(schema_names)
        return dict(zip(names, ss.spawn(len(names))))

    def render_schema(self, site_list, schema, plan, child_seq, model_f_lo,
                      model_f_hi, dur_s):
        """渲染一个 TS 档（源跨测点逐 bin 共享，synthetic_sites 同构输出）。

        返回 (ts_list, info)；info 含 n_bands/events/events_dropped。
        dur_s：合成总时长（秒）；全局时间轴 = 记录真实时间（TDateTime 差）。
        """
        if not plan.records:
            empty = [{'name': schema['name'],
                      'sample_rate': schema['sample_rate'],
                      'begins': [], 'continuous': schema['continuous'],
                      'channels': [[] for _ in range(5)]} for _ in site_list]
            return empty, {'n_bands': 0, 'events': 0, 'events_dropped': 0}
        n0 = plan.records[0].n_samples
        assert all(r.n_samples == n0 for r in plan.records), \
            'RecordPlan 等长假设不成立（变长记录走 M4 逐记录装配路径）'
        sr0 = plan.records[0].sample_rate
        freqs = np.fft.rfftfreq(n0, d=1.0 / sr0)
        df = sr0 / n0
        t_begin = plan.records[0].begin

        # 两条派生流：记录白噪（顺序）与包络/事件（顺序）
        rng_noise = np.random.default_rng(child_seq)
        rng_env = np.random.default_rng(
            np.random.SeedSequence(child_seq.generate_state(8)))

        # ── L2 带集 / 包络 / γ / 事件（全局时间轴，一次生成）────────────
        total_s = n0 * len(plan.records) / sr0
        f_lo_band = max(model_f_lo, 2.0 / total_s)
        f_hi_band = min(model_f_hi, 0.4 * sr0)
        centers = bands_mod.band_centers(f_lo_band, f_hi_band,
                                         self.band_octave) \
            if f_hi_band > f_lo_band else []
        centers = [c for c in centers if 0.35 * c >= 2.5 * df]  # 可表示守卫
        seg_chains, gamma_arr = {}, None
        if self.alpha2 > 0 and centers:
            for fc in centers:
                for _pi in range(2):
                    seg_chains[(fc, _pi)] = chain_arrays(make_chain(
                        rng_env, fc, dur_s, self.burst_periods,
                        self.burst_len_sigma,
                        _db_to_ln(self.burst_sigma_db)))
            if self.gamma_sigma_db > 0:
                gamma_arr = chain_arrays(make_chain(
                    rng_env, 1.0 / max(3600.0, dur_s / 8.0), dur_s,
                    max(1.0, dur_s / (8.0 * 3600.0)), 0.0,
                    _db_to_ln(self.gamma_sigma_db)))
        events, n_dropped = [], 0
        if self.alpha3 > 0 and self.event_rate > 0:
            events = make_events(rng_env, dur_s, max(model_f_lo, 2.0 / dur_s),
                                 min(model_f_hi, 0.4 * sr0), self.event_rate,
                                 self.event_q, self.event_sigma_db)
            # 占空比语义（§2.6）：与任一记录的真实时间窗相交才算被记录
            wins = [((r.begin - t_begin) * 86400.0,
                     (r.begin - t_begin) * 86400.0 + r.n_samples / sr0)
                    for r in plan.records]
            for ev in events:
                if not any(ev[0] >= a - 4 * ev[2] and ev[0] <= b + 4 * ev[2]
                           for a, b in wins):
                    n_dropped += 1

        # ── 每记录：L1 + L2 + L3 → 源谱 P̂（跨测点共享）→ 逐站渲染 ────
        # 三层均为每 Hz 单位谱密度口径：白噪的每样本方差取 fs（L1 直取、
        # L2 经带掩模继承、L3 事件以 √(α3·fs) 折算）——跨采样档重叠频段的
        # 物理谱密度一致（R2c② 合同；曾漏乘 √fs 致 TS3 的 L2 份额 ~4%）
        Ts = [render_fft.transfer_matrix(s, freqs) for s in site_list]
        amp_l1 = math.sqrt(self.alpha1 * sr0)
        amp_l2 = math.sqrt(self.alpha2 * sr0 / self._eg2)
        a_ref_ev = math.sqrt(self.alpha3 * sr0)
        masks = {c: bands_mod.band_amplitude(freqs, c, self.band_octave)
                 for c in centers}
        outs = [np.empty((5, n0 * len(plan.records)), dtype=np.float32)
                for _ in site_list]
        offs = np.arange(len(plan.records) + 1) * n0
        for j, rec in enumerate(plan.records):
            t_axis = (np.arange(n0) / sr0) + (rec.begin - t_begin) * 86400.0
            p1 = rng_noise.standard_normal(n0) * amp_l1
            p2 = rng_noise.standard_normal(n0) * amp_l1
            if self.alpha2 > 0 and centers:
                gref = sample_chain(gamma_arr[0], gamma_arr[1], t_axis, 0.0) \
                    if gamma_arr is not None else None
                for fc in centers:
                    mask = masks[fc]
                    eg, eh = seg_chains[(fc, 0)], seg_chains[(fc, 1)]
                    fade = CROSSFADE_PERIODS / fc
                    g = [sample_chain(eg[0], eg[1], t_axis, fade),
                         sample_chain(eh[0], eh[1], t_axis, fade)]
                    if gref is not None:
                        g[0] *= gref
                        g[1] *= gref
                    for pi in range(2):
                        w = rng_noise.standard_normal(n0)
                        nb = np.fft.irfft(mask * np.fft.rfft(w), n=n0)
                        if pi == 0:
                            p1 = p1 + (amp_l2 * g[0]) * nb
                        else:
                            p2 = p2 + (amp_l2 * g[1]) * nb
            if events:
                e1, e2 = _render_events(events, t_axis, a_ref_ev)
                p1 = p1 + e1
                p2 = p2 + e2
            P = [np.fft.rfft(p1), np.fft.rfft(p2)]
            P[0][0] = 0.0
            P[1][0] = 0.0
            for k in range(len(site_list)):
                outs[k][:, offs[j]:offs[j + 1]] = \
                    render_fft.render_site(P, Ts[k], n0)
        result = []
        for k, site in enumerate(site_list):
            channels = [[outs[k][c][offs[j]:offs[j + 1]]
                         for j in range(len(plan.records))] for c in range(5)]
            result.append({'name': schema['name'], 'sample_rate': sr0,
                           'begins': [r.begin for r in plan.records],
                           'continuous': plan.continuous,
                           'channels': channels})
        return result, {'n_bands': len(centers), 'events': len(events),
                        'events_dropped': n_dropped}
