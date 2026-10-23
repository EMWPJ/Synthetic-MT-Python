# -*- coding: utf-8 -*-
"""极简 MT 数据处理（参考 MTDataPro 处理链：读 TS → 加窗 FFT → 标定 → 阻抗估计）。

处理对象：SynthTSPy 合成的 Phoenix TS2-TS5 二进制时间序列。流程：
1. 逐频处理（频率 = 合成频率）：在包含该频率的 TS 文件中选窗数最多的一档，
   取窗口长 = 8 倍信号周期、汉宁窗、非重叠滑窗，对每个通道做单频 DFT
   X(f) = Σ w[n]·x[n]·e^{-2πi f n/fs}（等价于 MTDataPro 的逐频加窗 FFT 取目标频点）。
2. 标定：合成时 AddCalibration 做了 E_adc = E/rE、H_adc = H/rH·μ0，
   此处精确相乘还原物理场（与主工程 FFT 校准"谱除系统响应"同一环节；
   responds 用与合成相同的 calib.compute_responds，nearest-neighbour 完全一致）。
3. 阻抗估计（最小二乘）：对每个频率，把 N 个窗口的 (Hx,Hy) 作为输入矩阵、
   Ex/Ey 分别作为输出，解复数 LSQ  E = H·Z + ε，Z 行向量 = (H^H H)^{-1} H^H E。
   两个极化态在合成中带随机段幅相（等效随机源场），LSQ 跨窗口消去。

理论一致性：无噪声合成下，窗口 DFT 对每通道满足 X_ch = P1·C1_ch + P2·C2_ch
（P1/P2 为窗口相关的复权重、通道无关），LSQ 精确恢复 Z；残余误差仅来自
24bit 量化与汉宁窗对负频率像的旁瓣抑制（<-80dB）。
"""
import math
import os

import numpy as np

from . import calib
from .model import load_site
from .render_fft import interp_log_complex

MU0 = 4e-7 * math.pi

# 合成 schema 的各档频段与采样率（SynthTSPy.synth.build_schemas）
TS_SPEC = {
    'TS2': (10.0, 12000.0, 24000),
    'TS3': (1.0, 1200.0, 2400),
    'TS4': (0.1, 75.0, 150),
    'TS5': (1e-6, 7.5, 15),
}

WINDOW_PERIODS = 8.0   # 窗口长 = 8 倍信号周期（需求指定）


def read_ts(path):
    """读一个 .TSn 文件 → (sample_rate, data[5][n])，跨记录拼接（32B 头 + 24bit 小端）。"""
    raw = open(path, 'rb').read()
    chans = [[] for _ in range(5)]
    sr = None
    off = 0
    while off + 32 <= len(raw):
        head = raw[off:off + 32]
        scans = int.from_bytes(head[10:12], 'little')
        ch = head[12]
        if not (0 < scans <= 100000 and 0 < ch <= 8):
            break
        nbytes = ch * scans * 3
        body = raw[off + 32:off + 32 + nbytes]
        if len(body) < nbytes:
            break
        b = np.frombuffer(body, dtype=np.uint8).reshape(-1, 3)
        v = b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)
        v = np.where(v >= (1 << 23), v - (1 << 24), v)     # 符号扩展
        m = v.reshape(scans, ch).T.astype(float)           # [ch][scans]
        for c in range(min(ch, 5)):
            chans[c].append(m[c])
        sr = int.from_bytes(head[18:20], 'little')
        off += 32 + nbytes
    data = [np.concatenate(chans[c]) if chans[c] else np.zeros(0) for c in range(5)]
    return sr, data


def _kernel(fs, f, n):
    w = np.hanning(n)                                    # 0.5(1-cos(2πi/(N-1)))，同 MTDataPro HanningWin
    return w, w * np.exp(-2j * math.pi * f * np.arange(n) / fs)


def window_dft(x, fs, f, periods=WINDOW_PERIODS, overlap=0.5):
    """滑窗单频 DFT：窗长 = periods 个周期，汉宁窗，默认 50% 重叠（MT 惯例）。

    返回各窗口的复振幅数组 X_w = Σ w[n]·x[s+n]·e^{-2πi f n/fs}。
    """
    n = int(round(periods * fs / f))
    if n < 8 or len(x) < n:
        return np.zeros(0, dtype=complex)
    _, kernel = _kernel(fs, f, n)
    step = max(int(round(n * (1.0 - overlap))), 1)
    nwin = (len(x) - n) // step + 1
    out = np.empty(nwin, dtype=complex)
    for k in range(nwin):
        s = k * step
        out[k] = np.dot(x[s:s + n], kernel)
    return out


def screen_windows(xs, fs, f, periods=WINDOW_PERIODS, threshold=0.2):
    """半窗一致性筛查（MT 数据质量筛选惯例）。

    对每个窗同时算前/后半窗的单频 DFT（各 periods/2 周期）；源场平稳时两者
    复振幅一致，含相位跳变（分段边界）的窗差异大。返回通过筛查的窗口掩码，
    剔除跳变窗以抑制其负频率泄漏对 LSQ 的偏差。
    """
    n = int(round(periods * fs / f))
    half = n // 2
    if half < 8 or len(xs[0]) < n:
        return np.zeros(0, dtype=bool)
    w_half, _ = _kernel(fs, f, half)
    # 半窗核对齐各半窗中心（时间零点取窗中，避免相位基准漂移）
    kern_a = w_half * np.exp(-2j * math.pi * f * (np.arange(half) + (n - half) // 2) / fs)
    step = max(n // 2, 1)                                 # 与 window_dft 默认 50% 重叠同步
    nwin = (len(xs[0]) - n) // step + 1
    mask = np.zeros(nwin, dtype=bool)
    for k in range(nwin):
        s = k * step
        xa = np.dot(xs[0][s:s + half], kern_a)
        xb = np.dot(xs[0][s + n - half:s + n], kern_a)
        denom = abs(xa) + abs(xb)
        if denom < 1e-30:
            continue
        mask[k] = abs(xa - xb) / denom < threshold
    return mask


def pick_ts_file(freq, ts_dir, site_name):
    """在包含该频率的 TS 文件中选"窗数最多"的一档（连续档通常覆盖远多于分段的窄带档）。"""
    best = None
    for name, (fmin, fmax, fs) in TS_SPEC.items():
        if not (fmin < freq < fmax):
            continue
        # 每周期至少 4 样本：频段上缘(如 TS4 的 72Hz@150Hz 仅 2 spp)的档窗长
        # 只有 ~16 样本、频率分辨率崩坏——宁可少窗也选采样充裕的档
        # （曾致加密网格频段上缘 Z 误差 ~2）
        if fs / freq < 4.0:
            continue
        path = os.path.join(ts_dir, f'{site_name}.{name}')
        if not os.path.isfile(path):
            continue
        sr, data = read_ts(path)
        n = int(round(WINDOW_PERIODS * fs / freq))
        nwin = max((len(data[0]) - n) // n + 1, 0) if len(data[0]) >= n else 0
        if nwin > 0 and (best is None or nwin > best[1]):
            best = (path, nwin, sr, data)
    if best is None:
        return None
    return best[0], best[2], best[3], best[1]


def estimate_impedance_at(freq, ts_dir, site_name, bypass_screen=False):
    """单频阻抗估计（标定 + 复数 LSQ）。返回 (Z 2x2 complex, n_windows) 或 None。

    bypass_screen=True 跳过半窗一致性筛查（v2 噪声源用：独立复高斯半窗对
    的通过率仅 ~4%，筛查口径不适配；默认 False 行为与 v1 逐位一致）。
    """
    picked = pick_ts_file(freq, ts_dir, site_name)
    if picked is None:
        return None
    _, fs, data, nwin = picked
    xe = window_dft(data[0], fs, freq)
    xy = window_dft(data[1], fs, freq)
    xhx = window_dft(data[2], fs, freq)
    xhy = window_dft(data[3], fs, freq)

    # 标定：合成链 E_adc = E/rE、H_adc = H/rH·μ0 的精确逆
    r = calib.compute_responds([freq])
    fe = xe * r[0][0]
    fy = xy * r[1][0]
    fhx = xhx * r[2][0] / MU0
    fhy = xhy * r[3][0] / MU0

    # 半窗筛查：剔除含分段边界（相位跳变）的窗（噪声源可经 bypass_screen 跳过）
    if not bypass_screen:
        mask = screen_windows([data[0]], fs, freq)
        if mask.sum() >= 8 and len(mask) == len(fe):
            fe = fe[mask]; fy = fy[mask]; fhx = fhx[mask]; fhy = fhy[mask]

    # 复数 LSQ: [Hx Hy]·[Zxx Zxy]^T = Ex（Ey 同理）
    hmat = np.column_stack([fhx, fhy])
    zxx, zxy = np.linalg.lstsq(hmat, fe, rcond=None)[0]
    zyx, zyy = np.linalg.lstsq(hmat, fy, rcond=None)[0]
    return np.array([[zxx, zxy], [zyx, zyy]]), len(xe)


def theoretical_impedance(model_file):
    """从 XMLSITE 模型场反推理论阻抗：E = Z·H（两极化态），Z = E·H^{-1}。"""
    site = load_site(model_file)
    out = []
    for f in site.fields:
        if f.is_noise:
            continue
        hmat = np.array([[f.hx1, f.hx2], [f.hy1, f.hy2]])   # 行 Hx/Hy，列极化
        emat = np.array([[f.ex1, f.ex2], [f.ey1, f.ey2]])
        z = emat @ np.linalg.inv(hmat)
        out.append((f.frequency, z))
    return out


def process(model_file, ts_dir, site_name=None, bypass_screen=False):
    """端到端：合成频率逐频估计阻抗，与理论值比对。

    返回 [(freq, Z_est, Z_theory, nwin, max_rel_err)]。
    """
    site = load_site(model_file)
    if site_name is None:
        site_name = site.name or os.path.splitext(os.path.basename(model_file))[0]
    theory = dict(theoretical_impedance(model_file))
    results = []
    for f in site.fields:
        if f.is_noise:
            continue
        est = estimate_impedance_at(f.frequency, ts_dir, site_name,
                                    bypass_screen=bypass_screen)
        if est is None:
            continue
        z_est, nwin = est
        z_th = theory[f.frequency]
        err = np.abs(z_est - z_th) / np.maximum(np.abs(z_th), 1e-30)
        results.append((f.frequency, z_est, z_th, nwin, float(err.max())))
    return results


# ── 频域逐 bin 处理（v2.2，Hann 块窗 + 标定台阶感知）──────────────────
#
# 动机：8 周期单频 DFT 核（±0.32 oct）在连续谱上引入"Z 核加权平均"偏置。
# 渲染端保证逐 FFT bin E_c[k] = Σ_p T_{c,p}(f_k)·P̂_p[k]（线性代数精确），
# 处理端按窗×bin 堆叠样本、逐 bin 做 2×2 复 LSQ。理论参考用渲染同构口径
# （归一后锚点场各自 log 插值到 bin 频率再求 E·H⁻¹）。
#
# 块窗用 Hann（v2.2）：矩形块的 Dirichlet 远旁瓣（1/Δ 衰减）会把远处 CTS
# 最近邻标定台阶的能量按通道不同混入本 bin（E/H 台阶相位不同），实测
# p50 恶化 ~9 倍；Hann 主瓣仅 ±2 bin（核平均偏置在此宽度下可忽略），
# 首旁瓣 −31dB、18dB/oct 滚降，台阶混叠被压回主瓣邻域。
#
# CTS 台阶 bin：最近邻标定在网格中点处跳变（band5 相邻行相位差 25°~74°），
# 处理窗核跨中点混叠时无单值 Z 可估——cal_step_mask 给出这些 bin。


def cts_step_freqs(f_lo, f_hi, band=None):
    """CTS 最近邻标定的台阶频率（[f_lo, f_hi] 内）。

    含两类：各 band 网格相邻行中点，以及 band 交界（>400→2、>8→3、
    >0.8→4）处两套网格边缘行之间的切换阶跃。
    """
    out = []
    bands = (band,) if band else (2, 3, 4, 5)
    for b in bands:
        grid = calib._grid_for(b)
        if grid is None:
            continue
        gf = grid[0]
        for a, c in zip(gf[:-1], gf[1:]):
            m = 0.5 * (a + c)
            if f_lo <= m <= f_hi:
                out.append(float(m))
    for edge in (0.8, 8.0, 400.0):
        if f_lo <= edge <= f_hi:
            out.append(edge)
    return sorted(set(out))


def cal_step_mask(freqs, nfft, sr=None):
    """标定台阶 bin 掩码：True=应排除（距最近台阶中点 ≤ Hann 主瓣半宽）。"""
    freqs = np.asarray(freqs, dtype=float)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    half = 2.0 * df                                    # Hann 主瓣半宽 ±2 bin
    mask = np.zeros(len(freqs), dtype=bool)
    lo, hi = float(freqs[1]), float(freqs[-1])
    for m in cts_step_freqs(lo - half, hi + half):
        mask |= np.abs(freqs - m) <= half
    return mask


def read_ts_binned(ts_path, nfft, hop=None, window='hann'):
    """读 TS 文件按 nfft 分窗 rfft（hop 缺省=nfft 不重叠；nfft//2=50% 重叠）。

    window='hann'（缺省）抑制矩形块 Dirichlet 远旁瓣的标定台阶混叠；
    window='rect' 保留旧行为。
    返回 (freqs[nb], Ch[4][nb, n_win])，Ch 顺序 Ex/Ey/Hx/Hy（未标定计数）。
    """
    if hop is None:
        hop = nfft
    raw = open(ts_path, 'rb').read()
    chans = [[] for _ in range(5)]
    sr = None
    off = 0
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
        off += 32 + nbytes
    data = [np.concatenate(chans[c]) for c in range(5)]
    w = np.hanning(nfft) if window == 'hann' else np.ones(nfft)
    nwin = max((len(data[0]) - nfft) // hop + 1, 0)
    nb = nfft // 2 + 1
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    chout = []
    for c in range(4):
        acc = np.empty((nb, nwin), dtype=complex)
        for i in range(nwin):
            acc[:, i] = np.fft.rfft(
                data[c][i * hop:(i * hop + nfft)] * w)
        chout.append(acc)
    return freqs, chout, nwin


def process_fft_bins(model_file, ts_dir, band='TS5', nfft=1 << 10,
                     site_name=None, f_min=0.0, f_max=np.inf,
                     aggregate='lsq', hop=None, window='hann',
                     exclude_cal_steps=False):
    """频域逐 bin 阻抗估计（零核偏置）。

    aggregate='lsq'：跨窗堆叠样本做 2×2 复 LSQ（方差最优）。
    window='hann'（缺省）/ 'rect'；exclude_cal_steps=True 时剔除 CTS 台阶
    邻域 bin（最近邻标定中点 ±主瓣，无单值 Z 可估）。
    返回 [(f_bin, Z_est, Z_theory_render, n_win, rel_err_max)]，
    bin 频率限于 [f_min, f_max]。理论参考为渲染同构口径（见上）。
    """
    sname = site_name or 'MTA'
    path = os.path.join(ts_dir, '%s.%s' % (sname, band))
    freqs, ch, nwin = read_ts_binned(path, nfft, hop=hop, window=window)
    step_mask = cal_step_mask(freqs, nfft) if exclude_cal_steps \
        else np.zeros(len(freqs), dtype=bool)
    model = load_site(model_file)
    # 渲染同构的归一后锚点场（nature 归一为逐锚点列缩放：渲染端 T 在被
    # 不同常数缩放的锚点场之间插值，其隐含 Z ≠ "Z 值直接插值"，必须
    # 复刻同一缩放再插值求比）
    from .cli import nature_magnetic_amplitude
    rows_map = {x.frequency: x for x in model.fields if not x.is_noise}
    af = sorted(rows_map)
    scaled = {}
    for name in ('ex1', 'ey1', 'hx1', 'hy1', 'ex2', 'ey2', 'hx2', 'hy2'):
        v = []
        for a in af:
            row = rows_map[a]
            nat = nature_magnetic_amplitude(a)
            se = nat / math.sqrt(abs(row.hx1) ** 2 + abs(row.hy1) ** 2)
            sb = nat / math.sqrt(abs(row.hx2) ** 2 + abs(row.hy2) ** 2)
            k = 0 if name in ('ex1', 'ey1', 'hx1', 'hy1') else 1
            v.append(getattr(row, name) * (se if k == 0 else sb))
        scaled[name] = np.asarray(v)
    lf = np.log10(np.asarray(af, dtype=float))
    r = calib.responds_on_grid(freqs)
    lgb = np.log10(np.maximum(freqs, 1e-300))

    def field_at(name):
        return (np.interp(lgb, lf, scaled[name].real)
                + 1j * np.interp(lgb, lf, scaled[name].imag))

    E1, E2 = field_at('ex1'), field_at('ex2')
    H1, H2 = field_at('hx1'), field_at('hx2')
    EY1, EY2 = field_at('ey1'), field_at('ey2')
    HY1, HY2 = field_at('hy1'), field_at('hy2')
    det = H1 * HY2 - H2 * HY1                            # Hx_1·Hy_2 − Hx_2·Hy_1
    with np.errstate(divide='ignore', invalid='ignore'):
        Zxx_ref = (E1 * HY2 - E2 * HY1) / det
        Zxy_ref = (E2 * H1 - E1 * H2) / det
        Zyx_ref = (EY1 * HY2 - EY2 * HY1) / det
        Zyy_ref = (EY2 * H1 - EY1 * H2) / det

    out = []
    for k in range(1, len(freqs) - 1):
        f = freqs[k]
        if not (f_min <= f <= f_max) or step_mask[k]:
            continue
        hx = ch[2][k] * r[2][k] / MU0
        hy = ch[3][k] * r[3][k] / MU0
        ex = ch[0][k] * r[0][k]
        ey = ch[1][k] * r[1][k]
        hmat = np.column_stack([hx, hy])
        if hmat.shape[0] < 2 or np.linalg.matrix_rank(hmat) < 2:
            continue
        zxx, zxy = np.linalg.lstsq(hmat, ex, rcond=None)[0]
        zyx, zyy = np.linalg.lstsq(hmat, ey, rcond=None)[0]
        z = np.array([[zxx, zxy], [zyx, zyy]])
        zt = np.array([[Zxx_ref[k], Zxy_ref[k]],
                       [Zyx_ref[k], Zyy_ref[k]]])
        err = float((np.abs(z - zt) / np.maximum(np.abs(zt), 1e-30)).max())
        out.append((f, z, zt, nwin, err))
    return out


def kernel_corrected_theory(model_file, nfft, f_min, f_max, src_psd_grid,
                            grid_freqs, fs_band):
    """“理论过处理核”的预测：核平均 Z 的数值精确计算。

    处理窗（nfft 样本 Hann）的频域功率核 |W|^2 在 bin 网格上取值（数值
    DFT，非解析近似），核内 Z(f)^S_H 加权平均即为逐 bin LSQ 的期望输出
    （处理估计 vs 本预测的残差应仅剩量化与统计起伏）。f0 取模型锚点。
    """
    model = load_site(model_file)
    af = sorted(model.get_frequencies(False))
    rows_map = {x.frequency: x for x in model.fields if not x.is_noise}
    from .cli import nature_magnetic_amplitude
    # 渲染同构的归一后锚点场 → 核内积分网格上的场比
    grid = []
    for f0 in af:
        if not (f_min <= f0 <= f_max):
            continue
        tw = nfft / f0
        lo = max(f0 - 8.0 * f0 / nfft * 2, f_min * 0.5)
        hi = min(f0 + 8.0 * f0 / nfft * 2, f_max * 2)
        if hi <= lo:
            continue
        grid.extend(np.logspace(math.log10(lo), math.log10(hi), 161))
    grid = np.array(sorted(set(np.round(grid, 12))))
    cols = {}
    for name in ('ex1', 'ey1', 'hx1', 'hy1', 'ex2', 'ey2', 'hx2', 'hy2'):
        cols[name] = interp_log_complex(
            af, [getattr(rows_map[q], name) for q in af], grid)
    det = (cols['hx1'] * cols['hy2'] - cols['hx2'] * cols['hy1']) + 0j
    Zg = np.empty((len(grid), 2, 2), dtype=complex)
    with np.errstate(divide='ignore', invalid='ignore'):
        Zg[:, 0, 0] = (cols['ex1'] * cols['hy2'] - cols['ex2'] * cols['hy1']) / det
        Zg[:, 0, 1] = (cols['ex2'] * cols['hx1'] - cols['ex1'] * cols['hx2']) / det
        Zg[:, 1, 0] = (cols['ey1'] * cols['hy2'] - cols['ey2'] * cols['hy1']) / det
        Zg[:, 1, 1] = (cols['ey2'] * cols['hx1'] - cols['ey1'] * cols['hx2']) / det
    lg = np.log10(np.maximum(grid, 1e-300))
    lf_src = np.log10(np.maximum(np.asarray(grid_freqs, dtype=float), 1e-300))
    sgrid = np.interp(lg, lf_src, np.asarray(src_psd_grid, dtype=float))
    wspec = np.abs(np.fft.rfft(np.hanning(nfft)))
    wspec = wspec / wspec[1]
    nb = nfft // 2 + 1
    df_bin = fs_band / nfft
    out = []
    for f0 in af:
        if not (f_min <= f0 <= f_max):
            continue
        m = (grid >= f0 - nb * df_bin) & (grid <= f0 + nb * df_bin)
        if m.sum() < 3:
            continue
        kk = np.abs(grid[m] - f0) / df_bin
        ki = np.clip(np.round(kk).astype(int), 0, len(wspec) - 1)
        w = wspec[ki] * (np.abs(Zg[m][:, 0, 1]) ** 2) * sgrid[m]
        if w.sum() <= 0:
            continue
        zp = (Zg[m] * w[:, None, None]).sum(axis=0) / w.sum()
        out.append((f0, zp, Zg[m][int(np.argmin(np.abs(grid[m] - f0)))]))
    return out
