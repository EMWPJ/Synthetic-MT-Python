"""
时间序列处理器

提供:
- FFT频谱分析
- 功率谱密度计算
- 阻抗估算
- 视电阻率和相位计算
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class SpectrumResult:
    """频谱分析结果"""

    frequencies: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    power: np.ndarray
    n_samples: int
    sample_rate: float


class TimeSeriesProcessor:
    """
    时间序列处理器

    提供FFT分析、谱估计等功能
    """

    def __init__(
        self,
        ex: np.ndarray,
        ey: np.ndarray,
        hx: np.ndarray,
        hy: np.ndarray,
        hz: np.ndarray,
        sample_rate: float,
    ):
        """
        Args:
            ex, ey, hx, hy, hz: 电磁场时间序列
            sample_rate: 采样率 (Hz)
        """
        self.ex = ex
        self.ey = ey
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.sample_rate = sample_rate
        self.n = len(ex)
        self.duration = self.n / sample_rate

        # 预计算频率数组
        self._freqs = np.fft.fftfreq(self.n, 1.0 / sample_rate)

    def compute_fft(self, signal: np.ndarray, remove_dc: bool = True) -> SpectrumResult:
        """
        计算信号的FFT

        Args:
            signal: 输入信号
            remove_dc: 是否去除直流分量

        Returns:
            SpectrumResult对象
        """
        # 去除均值(直流分量)
        if remove_dc:
            signal = signal - np.mean(signal)

        # FFT
        fft_vals = np.fft.fft(signal)

        # 只取正频率 - apply mask to full FFT first, then extract
        pos_mask = self._freqs > 0
        freqs = self._freqs[pos_mask]
        fft_pos = fft_vals[pos_mask]

        # 归一化: 单边谱乘2除以n (双边谱的直流分量为n,非直流分量为n/2)
        amplitude = np.abs(fft_pos) * 2.0 / self.n
        phase = np.angle(fft_pos, deg=True)
        power = amplitude**2

        return SpectrumResult(
            frequencies=freqs,
            amplitude=amplitude,
            phase=phase,
            power=power,
            n_samples=self.n,
            sample_rate=self.sample_rate,
        )

    def compute_cross_spectrum(
        self, signal1: np.ndarray, signal2: np.ndarray
    ) -> np.ndarray:
        """
        计算互谱 (Cross spectrum)

        Gxy = Fx * conj(Fy)

        Args:
            signal1, signal2: 两个信号

        Returns:
            互谱数组 (complex)
        """
        # 去除直流
        s1 = signal1 - np.mean(signal1)
        s2 = signal2 - np.mean(signal2)

        # FFT
        F1 = np.fft.fft(s1)
        F2 = np.fft.fft(s2)

        # 互谱: G12 = F1 * conj(F2)
        return F1 * np.conj(F2)

    def estimate_impedance_fftw(self) -> Dict[str, np.ndarray]:
        """
        使用FFT和窗口法估算阻抗

        阻抗估算: Zxy ≈ Ex / Hy (在频率域)

        Returns:
            包含估算阻抗的字典
        """
        # 计算各分量的FFT
        ex_fft = np.fft.fft(self.ex - np.mean(self.ex))
        ey_fft = np.fft.fft(self.ey - np.mean(self.ey))
        hx_fft = np.fft.fft(self.hx - np.mean(self.hx))
        hy_fft = np.fft.fft(self.hy - np.mean(self.hy))

        # 只取正频率
        pos_mask = self._freqs > 0
        freqs = self._freqs[pos_mask]

        # 互功率谱
        ExHy = (ex_fft * np.conj(hy_fft))[pos_mask] * 2.0 / self.n
        EyHx = (ey_fft * np.conj(hx_fft))[pos_mask] * 2.0 / self.n
        HyHy = (hy_fft * np.conj(hy_fft))[pos_mask] * 2.0 / self.n
        HxHx = (hx_fft * np.conj(hx_fft))[pos_mask] * 2.0 / self.n

        # 阻抗估算 (最小二乘意义下)
        # Zxy = <ExHy> / <HyHy>
        # Zyx = <EyHx> / <HxHx>
        # 避免除零
        eps = 1e-20
        Zxy = ExHy / (HyHy + eps)
        Zyx = EyHx / (HxHx + eps)

        return {
            "frequencies": freqs,
            "Zxy": Zxy,
            "Zyx": Zyx,
            "coherence_xy": np.abs(ExHy) ** 2
            / (np.abs(ex_fft[pos_mask]) ** 2 * np.abs(hy_fft[pos_mask]) ** 2 + eps),
            "coherence_yx": np.abs(EyHx) ** 2
            / (np.abs(ey_fft[pos_mask]) ** 2 * np.abs(hx_fft[pos_mask]) ** 2 + eps),
        }

    def estimate_impedance_at_periods(
        self, periods: np.ndarray, freq_tol: float = 0.1
    ) -> Dict:
        """
        在指定周期处估算阻抗

        Args:
            periods: 目标周期数组 (s)
            freq_tol: 频率容差 (相对)

        Returns:
            包含周期、阻抗、视电阻率、相位的字典
        """
        impedance = self.estimate_impedance_fftw()
        freqs = impedance["frequencies"]

        target_freqs = 1.0 / periods
        n = len(periods)

        zxy_est = np.zeros(n, dtype=complex)
        zyx_est = np.zeros(n, dtype=complex)

        for i, target_f in enumerate(target_freqs):
            # 找到最近的频率
            idx = np.argmin(np.abs(freqs - target_f))
            if np.abs(freqs[idx] - target_f) / target_f < freq_tol:
                zxy_est[i] = impedance["Zxy"][idx]
                zyx_est[i] = impedance["Zyx"][idx]

        # 计算视电阻率和相位
        omega = 2 * np.pi / periods
        from .config import MU0

        rho_a = np.abs(zxy_est) ** 2 / (omega * MU0)
        phase = np.arctan2(zxy_est.imag, zxy_est.real) * 180.0 / np.pi

        return {
            "periods": periods,
            "frequencies": target_freqs,
            "Zxy": zxy_est,
            "Zyx": zyx_est,
            "app_resistivity": rho_a,
            "phase": phase,
        }

    def get_frequency_resolution(self) -> float:
        """获取频率分辨率"""
        return self.sample_rate / self.n

    def get_period_resolution(self) -> float:
        """获取周期分辨率"""
        return 1.0 / self.get_frequency_resolution()


def compute_periodogram(
    signal: np.ndarray, sample_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算周期图 (Periodogram)

    简单的功率谱估计方法

    Args:
        signal: 输入信号
        sample_rate: 采样率

    Returns:
        (frequencies, power_spectrum)
    """
    n = len(signal)
    signal = signal - np.mean(signal)

    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, 1.0 / sample_rate)

    pos_mask = freqs > 0
    power = np.abs(fft_vals[pos_mask]) ** 2 / (sample_rate * n)

    return freqs[pos_mask], power


def welch_method(
    signal: np.ndarray, sample_rate: float, nperseg: int = 256, noverlap: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welch方法 (改进的周期图)

    将信号分段，加窗后平均以降低方差

    Args:
        signal: 输入信号
        sample_rate: 采样率
        nperseg: 每段点数
        noverlap: 重叠点数

    Returns:
        (frequencies, power_spectrum)
    """
    if noverlap is None:
        noverlap = nperseg // 2

    n = len(signal)
    signal = signal - np.mean(signal)

    # 分段
    n_segments = (n - noverlap) // (nperseg - noverlap)

    # 汉宁窗
    window = np.hanning(nperseg)
    power_sum = np.zeros(nperseg // 2)

    for i in range(n_segments):
        start = i * (nperseg - noverlap)
        segment = signal[start : start + nperseg] * window

        fft_vals = np.fft.fft(segment)
        power_sum += np.abs(fft_vals[: nperseg // 2]) ** 2

    power = power_sum / (n_segments * nperseg * np.sum(window**2) / nperseg)
    freqs = np.fft.fftfreq(nperseg, 1.0 / sample_rate)[: nperseg // 2]

    return freqs[freqs > 0], power[freqs > 0]
