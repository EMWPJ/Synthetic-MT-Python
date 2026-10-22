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
