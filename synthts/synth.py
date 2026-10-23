# -*- coding: utf-8 -*-
"""合成核心：RandomSegmentPartiallyWindowed 单算法 + MT/AMT schema + 同源批量合成。

自 Delphi 移植而来，但不再追求与 MTDataPro 逐位一致——验收标准为"合成数据
经 mtprocess 处理后与理论阻抗一致"。

**同源批量**：随机性全部集中在"源场方案"(scenario)——每频点两极化的随机
分段(段长/幅值/相位)，与测点场无关。先对一批测点统一抽签一次，再逐测点用
各自的场向量渲染：相同频点在所有测点得到**完全一致的段结构与随机因子**
(模拟同一源场激励多测点，即远参考观测场景)；测点间的差异只来自各自的
标定后的场(阻抗/极化)。随机源 = numpy Generator 显式传参，同 -seed 可复现。

段长语义：段长 ≈ Ceil(N(sr/f·Normal(periods, periods/2)))，即平均段长
= periods 个信号周期。

**渲染性能版（float32 向量化 + 跨测点载波复用）**：
- 一个 schema 的全部时段拼接成整轴一次渲染（段起点全局化，段内局部序号
  int32 精确差分，回避大样本序号 f32 相位损失）；
- 每频每极化的载波 cos/sin（含段幅相与边界过渡窗增益）只算一遍，全部
  测点共享；测点差异仅体现为 (5,2)@(2,n) 的通道组合系数（sgemm）；
- f32 精度：段内相位 < ~2π·1.5·periods rad（ulp ~1e-5 rad），幅值相对
  误差 ~1e-7，远低于 24bit 量化分辨率与 Z 估计 1e-2 验收阈。
"""
import math

import numpy as np

TWO_PI = 2.0 * math.pi

# 各档频段与采样率（TS 文件规格）
TS2_FRE = (10.0, 12000.0)
TS3_FRE = (1.0, 1200.0)
TS4_FRE = (0.1, 75.0)
TS5_FRE = (1e-6, 7.5)
# AMT 无 TS5 档，连续 TS4 须覆盖到 0.01 Hz（MT 的分段 TS4 仍为 0.1 Hz 下限）
TS4_FRE_AMT = (0.01, 75.0)

_SECONDS_PER_DAY = 86400.0


def inv_hanning_win(count):
    """反转汉宁窗（段边界处的 1->0->1 过渡窗，跨边界平滑），f32。"""
    i = np.arange(count, dtype=np.float64)
    w = 0.5 * (1.0 - np.cos(TWO_PI * i / max(count - 1, 1)))
    return np.roll(w, count // 2).astype(np.float32)


# ── 源场方案（随机性集中地，与测点无关）─────────────────────────────────────

def draw_band_segments(sr, n_samples, freqs, periods, rng):
    """对一个 TS 档时段抽签源场方案。

    返回 {fre: [segs_pol1, segs_pol2]}，segs = [(start, count, da, dph), ...]。
    极化 1 段幅值 Normal(1,1)、极化 2 段幅值 Uniform(0,2)，段相位 Uniform(0,2π)；
    段长 Normal(periods, periods/2) 个周期（最短 10 样本，尾段并尾）。
    """
    scen = {}
    for fre in freqs:
        pols = []
        for series in (1, 2):
            segs = []
            leftcount = n_samples
            while leftcount > 0:
                tmpcount = int(math.ceil(abs(sr / fre * rng.normal(periods, periods / 2.0))))
                while tmpcount < 10:
                    tmpcount = int(math.ceil(abs(sr / fre * rng.normal(periods, periods / 2.0))))
                if series == 1:
                    da = rng.normal(1.0, 1.0)
                else:
                    da = rng.uniform(0.0, 2.0)
                dph = rng.uniform(0.0, TWO_PI)
                if tmpcount > leftcount:
                    tmpcount = leftcount
                if leftcount - tmpcount < 10:
                    tmpcount = leftcount
                segs.append((n_samples - leftcount, tmpcount, da, dph))
                leftcount -= tmpcount
            pols.append(segs)
        scen[fre] = pols
    return scen


def schema_segments(schema, btime, etime):
    """schema 的时段划分：[(begin_tdatetime, n_samples), ...]。"""
    sr = schema['sample_rate']
    if schema['continuous']:
        begins = [btime]
        durations = [(etime - btime) * _SECONDS_PER_DAY + 1.0]   # 秒
    else:
        total_s = int((etime - btime) * _SECONDS_PER_DAY)
        n_seg = (total_s - schema['offset_s']) // schema['interval_s'] + 1
        begins = [btime + (schema['interval_s'] * i + schema['offset_s'])
                  / _SECONDS_PER_DAY for i in range(n_seg)]
        durations = [float(schema['seg_s'])] * n_seg               # 秒
    return [(begins[k], int(round(durations[k] * sr))) for k in range(len(begins))]


def build_schema_scenarios(schemas, union_freqs, btime, etime, periods, rng):
    """对一批测点统一构建各 TS 档的源场方案（同源核心）。

    union_freqs：全体测点信号频点并集。同一档内按频率排序逐个抽签，
    各测点同频点共享同一方案条目。
    返回 {schema_name: [(begin, n, scen_dict), ...]}。
    """
    out = {}
    for schema in schemas:
        segs = schema_segments(schema, btime, etime)
        band = sorted(f for f in union_freqs
                      if schema['min_fre'] < f < schema['max_fre'])
        out[schema['name']] = [
            (begin, n, draw_band_segments(schema['sample_rate'], n, band,
                                          periods, rng))
            for begin, n in segs]
    return out


# ── 渲染（确定性：方案 + 测点场 -> 时间序列）───────────────────────────────

def _taper_gain(starts, counts, n_total, wcount, win, local_starts=None):
    """段边界过渡窗增益数组 g[n] = 各段窗系数之积（向量化散射累乘）。

    语义：count <= half 的段不加窗；local_start - half < 0 的段整体不加窗
    （时段首段，窗不得越过时段边界涂改相邻时段）；窗越过全局 [0, n) 的
    样本跳过；相邻段窗重叠处取乘积（乘法可交换，与施加次序无关）。
    """
    g = np.ones(n_total, dtype=np.float32)
    half = wcount // 2
    base = local_starts if local_starts is not None else starts
    active = (counts > half) & (base - half >= 0)
    st = starts[active]
    if len(st) == 0:
        return g
    off = np.arange(wcount, dtype=np.int64)
    idx = st[:, None] - half + off[None, :]           # (nact, wcount)
    ok = idx < n_total
    np.multiply.at(g, idx[ok], np.broadcast_to(win, idx.shape)[ok])
    return g


def _carrier(sr, fre, segs_global, n_total, M, w0, win, wcount):
    """计算一个频点一个极化的载波（含段幅相/边界过渡窗），写入 M (2, n) f32。

    segs_global: [(global_start, count, da, dph, local_start), ...]（本 schema
    全部时段拼接后的全局段表，local_start 为段在其所属时段内的起点）。
    段内局部序号 = 全局序号 - 段起点，保证 f32 相位精度：局部序号 <
    periods·1.5·sr/f，相位上限 ~2π·1.5·periods。
    窗激活条件用局部坐标（local_start - half >= 0）：时段首段不加窗（时段
    边界处各段独立渲染，窗不得越界涂改相邻时段的尾部样本）。
    """
    arr = np.array(segs_global, dtype=np.float64)
    starts = arr[:, 0].astype(np.int64)
    counts = arr[:, 1].astype(np.int64)
    local = arr[:, 4].astype(np.int64)
    ti = np.arange(n_total, dtype=np.int32)
    ti -= np.repeat(starts.astype(np.int32), counts)   # 段内局部序号（精确）
    tf = ti.astype(np.float32)
    del ti
    tf *= np.float32(w0)
    tf += np.repeat(arr[:, 3].astype(np.float32), counts)
    np.cos(tf, out=M[0])
    np.sin(tf, out=M[1])
    del tf
    gg = _taper_gain(starts, counts, n_total, wcount, win, local)
    gg *= np.repeat(arr[:, 2].astype(np.float32), counts)
    M[0] *= gg
    M[1] *= gg


def synthetic_sites(site_list, schema, scenario, rot=(0.0, 0.0, 0.0, 0.0)):
    """按共享源场方案渲染一批测点的一个 TS 档（载波跨测点复用）。

    site_list：已标定的测点列表。返回与 site_list 等长的 ts dict 列表：
    dict(name/sample_rate/begins/continuous/channels[5][segs][samples])。
    模型的 isNoise 频点不参与合成；测点未含的方案频点自动跳过。
    """
    sr = schema['sample_rate']
    preps = []
    for site in site_list:
        fre_cols = site.get_feh(noise=False)
        fre_list = fre_cols[0]
        f10 = [list(col) for col in fre_cols[1:]]
        _rotate_fields(f10, rot)
        preps.append(({f: i for i, f in enumerate(fre_list)}, f10))

    begins = [begin for begin, _, _ in scenario]
    ns = [n for _, n, _ in scenario]
    total = int(sum(ns))
    offs = np.concatenate(([0], np.cumsum(ns))).astype(np.int64)
    outs = [np.zeros((5, total), dtype=np.float32) for _ in site_list]
    M = np.empty((2, total), dtype=np.float32)
    buf = np.empty((5, total), dtype=np.float32)

    freqs = sorted(set().union(*[set(scen) for _, _, scen in scenario]))
    for fre in freqs:
        w0 = TWO_PI * fre / sr
        # 段边界过渡窗宽（约 1 周期）；下限 8 样本——高频点(f 接近 fs/4)按周期算
        # 只有 2 样本，inv_hanning(2)=[0,0] 会把边界样本挖零(制造跳变而非平滑)，
        # 曾致加密网格频段上缘误差 ~2e-2
        wcount = max(int(sr / 2.0 / fre) * 2, 8)
        win = inv_hanning_win(wcount)
        for series in (1, 2):
            segs_global = []
            for k, (_, _, scen) in enumerate(scenario):
                segs_global.extend((offs[k] + s, c, a, p, s)
                                   for s, c, a, p in scen[fre][series - 1])
            _carrier(sr, fre, segs_global, total, M, w0, win, wcount)
            for k, (idx, f10) in enumerate(preps):
                if fre not in idx:
                    continue
                base_idx = (series - 1) * 5
                PQ = np.empty((5, 2), dtype=np.float32)
                for c in range(5):
                    z = complex(f10[base_idx + c][idx[fre]])
                    amp = abs(z) * schema['source_scale']
                    ang = math.atan2(z.imag, z.real)
                    PQ[c, 0] = amp * math.cos(ang)
                    PQ[c, 1] = -amp * math.sin(ang)
                np.matmul(PQ, M, out=buf)
                outs[k] += buf

    result = []
    for k in range(len(site_list)):
        channels = [[outs[k][c][offs[j]:offs[j + 1]] for j in range(len(ns))]
                    for c in range(5)]
        result.append({'name': schema['name'], 'sample_rate': sr,
                       'begins': begins, 'continuous': schema['continuous'],
                       'channels': channels})
    return result


def synthetic_site(site, schema, scenario, rot=(0.0, 0.0, 0.0, 0.0)):
    """单测点便捷包装（synthetic_sites 的单元素调用）。"""
    return synthetic_sites([site], schema, scenario, rot)[0]


def _rotate_fields(f10, rot):
    """场极化旋转（每频点 8 个复量按角 exr/eyr/hxr/hyr 旋转）。"""
    exr, eyr, hxr, hyr = rot

    def fa(ax, ay, th):
        return ax * math.cos(th) + ay * math.sin(th)

    for i in range(len(f10[0])):
        v = [f10[c][i] for c in range(10)]
        for base in (0, 5):
            ex, ey, hx, hy = v[base], v[base + 1], v[base + 2], v[base + 3]
            v[base] = fa(ex, ey, exr)
            v[base + 1] = fa(ex, ey, exr + math.pi / 2)
            v[base + 2] = fa(hx, hy, hxr)
            v[base + 3] = fa(hx, hy, hxr + math.pi / 2)
        for c in range(10):
            f10[c][i] = v[c]


def build_schemas(tbl, mode, periods, scale):
    """构建 TS 档 schema（频段界与分段参数沿用 MTDataPro 约定）。

    MT -> TS3/TS4/TS5；AMT -> TS2/TS3/TS4（TS4 连续档）。
    """
    schemas = []

    def add(name, continuous, sr, fre_range, interval_s, offset_s, seg_s):
        schemas.append({
            'name': name, 'continuous': continuous, 'sample_rate': sr,
            'min_fre': fre_range[0], 'max_fre': fre_range[1],
            'interval_s': interval_s, 'offset_s': offset_s, 'seg_s': seg_s,
            'source_scale': scale, 'periods': periods,
        })

    if mode == 'AMT':
        add('TS2', False, 24000, TS2_FRE, tbl.hsmp * 2, tbl.hsmp, tbl.l2ns / 10.0)
        add('TS3', False, 2400, TS3_FRE, tbl.hsmp * 2, 0, float(tbl.l3ns))
        add('TS4', True, 150, TS4_FRE_AMT, 0, 0, 0.0)
    else:
        add('TS3', False, 2400, TS3_FRE, tbl.hsmp * 2, tbl.hsmp, float(tbl.l3ns))
        add('TS4', False, 150, TS4_FRE, tbl.hsmp * 2, 0, float(tbl.l4ns))
        add('TS5', True, 15, TS5_FRE, 0, 0, 0.0)
    return schemas
