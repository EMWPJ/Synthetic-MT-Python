# -*- coding: utf-8 -*-
"""v2 三层源（L1/L2/L3 + 可选谱形/昼夜调制/L4 加性噪声）。

**跨档共享源结构**（v2.1，科学文档 §8 局限①修复）：L2 包络段链、γ 公共
因子与 L3 事件提升为**全局 rng 流**（SeedSequence 顶层派生，按带/极化键
控）——突发时刻与事件在所有采样档是同一物理实现；带限载波白噪仍按
"schema×记录"独立抽签（同一连续过程的独立离散采样，间隙≫带相关时间时
统计等价，§3.3）。rng 层级：

    SeedSequence(seed)
     ├─ env   → 段链（带×极化）/ γ / 事件（全局时间轴，全档共享）
     ├─ l4    → 加性噪声（逐记录）
     └─ schema（按名称排序）→ 记录白噪流（L1+L2 载波）

功率归一化：三层均按每 Hz 单位谱密度口径（每样本方差 = fs）；L2 权重 α2
预除 E[g²γ²]（解析 exp(2σ²)），锚定均值 PSD。带集下截（带宽 <2.5 bin 的
带跳过，跨档由 pick_ts_file 覆盖）。
"""
import math

import numpy as np

from . import bands as bands_mod
from . import render_fft

LN_OF_DB = 8.685889638065035      # 10/ln(10)：dB ↔ ln 换算
MIN_BURST_PERIODS = 10.0          # 段长下限（§2.5 谱展宽守卫）
CROSSFADE_PERIODS = 16.0          # 段间升余弦过渡宽（§2.1）
DIURNAL_PERIOD_S = 86400.0        # 昼夜调制周期
# Schumann 谐振（磁径向分量，频率/品质因数量级取自观测文献）
SCHUMANN_PEAKS = ((7.83, 4.6), (14.3, 4.6), (20.8, 4.6), (26.4, 4.6))


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
    """段链 → (edges, gains) 数组（采样热路径预转换）。"""
    return (np.array([s[0] for s in segs] + [segs[-1][1]]),
            np.array([s[2] for s in segs]))


def sample_chain(edges, gains, t_axis, fade_s):
    """段链在真实时间轴 t_axis（秒，单调升）上的采样：分段常数 + 升余弦。

    过渡按边界二分切片施加（切片总长 ≤ n + 边界区少量重叠）。
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
    """L3 事件表（全局）：时间域 Poisson 到达 + log-uniform 中心频 + LN 幅度。"""
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


def _render_events(events, t_axis, a_ref, f_max):
    """事件在时间轴上的两极化贡献（源单位）；f_max 以上的事件不渲染（欠采样）。"""
    p1 = np.zeros(len(t_axis), dtype=np.float64)
    p2 = np.zeros(len(t_axis), dtype=np.float64)
    for t0, fc, tau, amp, phase, theta in events:
        if fc > f_max:
            continue
        dt = t_axis - t0
        m = np.abs(dt) <= 4.0 * tau
        if not m.any():
            continue
        wav = amp * np.exp(-0.5 * (dt[m] / tau) ** 2) \
            * np.cos(2.0 * math.pi * fc * dt[m] + phase)
        p1[m] += wav * math.cos(theta) * a_ref
        p2[m] += wav * math.sin(theta) * a_ref
    return p1, p2


def schumann_shape(freqs):
    """Schumann 谐振谱形（源侧乘子，返回幅度谱 √S）。"""
    f = np.asarray(freqs, dtype=float)
    s = np.ones_like(f)
    for f_i, q_i in SCHUMANN_PEAKS:
        hw = f_i / (2.0 * q_i)
        s = s + 1.0 * hw ** 2 / ((f - f_i) ** 2 + hw ** 2)
    return np.sqrt(s)


def diurnal_gain(t_axis, level_db):
    """昼夜调制 D(t) = 10^((dB/20)·½(1−cos(2πt/86400)))，峰值在 t=T/2。"""
    return 10.0 ** ((level_db / 20.0)
                    * 0.5 * (1.0 - np.cos(2.0 * math.pi * t_axis
                                          / DIURNAL_PERIOD_S)))


def add_l4_noise(X, rng, fs, t_axis, l4_type, level_db):
    """L4 加性噪声（逐通道、与源独立；level 相对**该记录该通道 RMS**）。

    white：高斯；impulsive：稀疏尖峰（泊松到达、重尾幅度）。
    不建模工频谐波——实测 Phoenix 数据经仪器前端陷波，工频已不在数据中
    （作者确认，2026-10-22）。
    """
    if l4_type == 'none' or level_db <= -120.0:
        return X
    amp = 10.0 ** (level_db / 20.0)
    out = X.astype(np.float64)
    for c in range(X.shape[0]):
        sig = float(np.sqrt(np.mean(out[c] ** 2)))
        if sig <= 0.0:
            continue
        sigma = sig * amp
        if l4_type == 'white':
            out[c] += rng.standard_normal(len(out[c])) * sigma
        elif l4_type == 'impulsive':
            rate = max(len(out[c]) / 5000.0, 1e-6)
            n = int(rng.poisson(rate))
            for _ in range(n):
                i = int(rng.integers(0, len(out[c])))
                out[c][i] += sigma * 10.0 * rng.standard_normal()
    return out.astype(np.float32)


class NoiseSource:
    """三层噪声源 + 可选谱形/昼夜调制/L4 加性噪声（设计 §2.1）。"""

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
        self.schumann = float(params.get('schumann', 0.0))
        self.diurnal_db = float(params.get('diurnal_db', 0.0))
        self.l4_type = str(params.get('l4_type', 'none'))
        self.l4_level_db = float(params.get('l4_level_db', -60.0))
        self._eg2 = math.exp(2.0 * _db_to_ln(self.burst_sigma_db) ** 2) \
            * math.exp(2.0 * _db_to_ln(self.gamma_sigma_db) ** 2)

    # ── 全局源结构（跨档共享：段链/γ/事件，v2.1）────────────────────────
    def prepare(self, model_f_lo, model_f_hi, dur_s, sr_max):
        """一次抽签全局源结构；各档渲染共享（突发时刻/事件跨档一致）。"""
        ss = np.random.SeedSequence(self.seed)
        g_env, g_l4 = ss.spawn(2)
        rng_env = np.random.default_rng(g_env)
        f_lo_g = max(model_f_lo, 2.0 / dur_s)
        f_hi_g = min(model_f_hi, 0.4 * sr_max)
        centers_g = bands_mod.band_centers(f_lo_g, f_hi_g, self.band_octave) \
            if f_hi_g > f_lo_g else []
        chains = {}
        if self.alpha2 > 0 and centers_g:
            for fc in centers_g:
                for pi in range(2):
                    chains[(fc, pi)] = chain_arrays(make_chain(
                        rng_env, fc, dur_s, self.burst_periods,
                        self.burst_len_sigma,
                        _db_to_ln(self.burst_sigma_db)))
        gamma_arr = None
        if self.alpha2 > 0 and self.gamma_sigma_db > 0:
            gamma_arr = chain_arrays(make_chain(
                rng_env, 1.0 / max(3600.0, dur_s / 8.0), dur_s,
                max(1.0, dur_s / (8.0 * 3600.0)), 0.0,
                _db_to_ln(self.gamma_sigma_db)))
        events, dropped = [], 0
        if self.alpha3 > 0 and self.event_rate > 0:
            events = make_events(rng_env, dur_s, f_lo_g, f_hi_g,
                                 self.event_rate, self.event_q,
                                 self.event_sigma_db)
        return {'env_seq': g_env, 'l4_seq': g_l4, 'centers': centers_g,
                'chains': chains, 'gamma': gamma_arr, 'events': events,
                'events_dropped': dropped}

    def schema_children(self, schema_names):
        """schema 白噪流子序列（按名称排序；全局流独立于此，互不影响）。"""
        ss = np.random.SeedSequence(self.seed + 1)
        names = sorted(schema_names)
        return dict(zip(names, ss.spawn(len(names))))

    def render_schema(self, site_list, schema, plan, schema_seq, state,
                      dur_s, record_wins):
        """渲染一个 TS 档。返回 (ts_list, info)。"""
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
        rng_noise = np.random.default_rng(schema_seq)
        rng_l4 = np.random.default_rng(state['l4_seq'])

        centers = [c for c in state['centers'] if 0.35 * c >= 2.5 * df
                   and c <= 0.4 * sr0]
        seg_chains = state['chains']
        gamma_arr = state['gamma']

        # 昼夜调制门控：dur<24h 不启用（设计 §2.7 R1 修正）
        diurnal_on = self.diurnal_db > 0 and dur_s >= DIURNAL_PERIOD_S
        shape = schumann_shape(freqs) if self.schumann > 0 else None

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
                # 带间共享同一白噪（相干带合成，v2.1）：交点处两带成分
                # 完全相干，混合平滑——独立实现会使带交点 bin 的有效 Z
                # 随两带突发权重随机摆动（曾致带交点 bin ~12% 偏差）
                for fc in centers:
                    mask = masks[fc]
                    eg, eh = seg_chains[(fc, 0)], seg_chains[(fc, 1)]
                    fade = CROSSFADE_PERIODS / fc
                    g = [sample_chain(eg[0], eg[1], t_axis, fade),
                         sample_chain(eh[0], eh[1], t_axis, fade)]
                    if gref is not None:
                        g[0] *= gref
                        g[1] *= gref
                    w = rng_noise.standard_normal(n0)
                    spec = np.fft.rfft(w)
                    nb1 = np.fft.irfft(mask * spec, n=n0)
                    w2 = rng_noise.standard_normal(n0)   # 极化独立白噪
                    spec2 = np.fft.rfft(w2)
                    nb2 = np.fft.irfft(mask * spec2, n=n0)
                    p1 = p1 + (amp_l2 * g[0]) * nb1
                    p2 = p2 + (amp_l2 * g[1]) * nb2
            if state['events']:
                e1, e2 = _render_events(state['events'], t_axis, a_ref_ev,
                                        0.4 * sr0)
                p1 = p1 + e1
                p2 = p2 + e2
            if diurnal_on:
                dg = diurnal_gain(t_axis, self.diurnal_db)
                p1 *= dg
                p2 *= dg
            P = [np.fft.rfft(p1), np.fft.rfft(p2)]
            if shape is not None:
                P[0] *= shape
                P[1] *= shape
            P[0][0] = 0.0
            P[1][0] = 0.0
            for k in range(len(site_list)):
                X = render_fft.render_site(P, Ts[k], n0)
                if self.l4_type != 'none':
                    X = add_l4_noise(X, rng_l4, sr0, t_axis,
                                     self.l4_type, self.l4_level_db)
                outs[k][:, offs[j]:offs[j + 1]] = X
        # 占空比：全局事件未落入任何档任何记录窗（±4τ）才算 dropped
        n_dropped = 0
        for ev in state['events']:
            if not any(ev[0] >= a - 4 * ev[2] and ev[0] <= b + 4 * ev[2]
                       for a, b in record_wins):
                n_dropped += 1
        result = []
        for k, site in enumerate(site_list):
            channels = [[outs[k][c][offs[j]:offs[j + 1]]
                         for j in range(len(plan.records))] for c in range(5)]
            result.append({'name': schema['name'], 'sample_rate': sr0,
                           'begins': [r.begin for r in plan.records],
                           'continuous': plan.continuous,
                           'channels': channels})
        return result, {'n_bands': len(centers),
                        'events': len(state['events']),
                        'events_dropped': n_dropped}
