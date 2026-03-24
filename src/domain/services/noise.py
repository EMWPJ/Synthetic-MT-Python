"""Domain service - Noise injection for time series.

This module contains pure business logic for adding various types of noise
to magnetotelluric time series. No external dependencies (no file I/O, no network).
"""

from typing import Tuple, Optional

import numpy as np

from ..value_objects import NoiseType, NoiseConfig


class NoiseInjector:
    """噪声注入器 - 用于向时间序列添加各种类型噪声"""

    def __init__(self, config: NoiseConfig, sample_rate: float, seed: Optional[int] = None):
        self.config = config
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

    def add_noise(self, *channels: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        向一个或多个通道添加噪声

        Parameters:
            *channels: 可变数量的时间序列数组

        Returns:
            添加噪声后的时间序列元组
        """
        noise_funcs = {
            NoiseType.SQUARE_WAVE: self._square_wave,
            NoiseType.TRIANGULAR: self._triangular,
            NoiseType.IMPULSIVE: self._impulsive,
            NoiseType.GAUSSIAN: self._gaussian,
            NoiseType.POWERLINE: self._powerline,
        }

        noise_func = noise_funcs.get(self.config.noise_type, self._gaussian)
        return tuple(noise_func(ch.copy()) for ch in channels)

    def _square_wave(self, data: np.ndarray) -> np.ndarray:
        """方波噪声"""
        if self.config.amplitude <= 0:
            return data

        n = len(data)
        if self.config.frequency > 0:
            period = int(self.sample_rate / self.config.frequency)
            if period > 0:
                t = np.arange(n) % period
                wave = np.where(t < period / 2, self.config.amplitude, -self.config.amplitude)
                noise = wave
            else:
                noise = np.zeros(n)
        else:
            noise = self.rng.choice([self.config.amplitude, -self.config.amplitude], n)

        return data + noise

    def _triangular(self, data: np.ndarray) -> np.ndarray:
        """三角波噪声"""
        if self.config.amplitude <= 0:
            return data

        n = len(data)
        if self.config.frequency > 0:
            period = int(self.sample_rate / self.config.frequency)
            if period > 0:
                t = np.arange(n) % period
                phase = t / period
                wave = np.where(phase < 0.5,
                              2 * self.config.amplitude * phase,
                              2 * self.config.amplitude * (1 - phase))
                noise = wave - self.config.amplitude / 2
            else:
                noise = np.zeros(n)
        else:
            noise = np.zeros(n)

        return data + noise

    def _impulsive(self, data: np.ndarray) -> np.ndarray:
        """冲击噪声（随机尖峰）"""
        if self.config.amplitude <= 0:
            return data

        n = len(data)
        mask = self.rng.random(n) < self.config.probability
        noise = np.zeros(n)
        n_spikes = int(np.sum(mask))
        if n_spikes > 0:
            noise[mask] = self.rng.choice([-1, 1], n)[mask] * \
                          self.config.amplitude * (0.5 + 0.5 * self.rng.random(n_spikes))

        return data + noise

    def _gaussian(self, data: np.ndarray) -> np.ndarray:
        """高斯噪声"""
        if self.config.amplitude <= 0:
            return data

        n = len(data)
        noise = self.rng.normal(0, self.config.amplitude, n)

        return data + noise

    def _powerline(self, data: np.ndarray) -> np.ndarray:
        """工频干扰 (50/60Hz)"""
        if self.config.amplitude <= 0 or self.config.frequency <= 0:
            return data

        n = len(data)
        t = np.arange(n, dtype=np.float64) / self.sample_rate
        noise = self.config.amplitude * np.sin(
            2 * np.pi * self.config.frequency * t + self.config.phase
        )

        return data + noise


def add_powerline_interference(data: np.ndarray, sample_rate: float,
                                frequency: float = 50.0, amplitude: float = 1.0,
                                phase: float = 0.0) -> np.ndarray:
    """
    添加工频干扰

    Parameters:
        data: 输入时间序列
        sample_rate: 采样率 (Hz)
        frequency: 工频频率 (Hz), 50或60
        amplitude: 干扰幅度
        phase: 初相位 (rad)

    Returns:
        添加干扰后的时间序列
    """
    n = len(data)
    t = np.arange(n, dtype=np.float64) / sample_rate
    interference = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return data + interference