"""
SyntheticMT - 大地电磁合成时间序列

基于论文: Wang P, Chen X, Zhang Y (2023) Synthesizing magnetotelluric 
time series based on forward modeling. Front. Earth Sci. 11:1086749

设计原则：简洁、Pythonic、模块化
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence
from enum import Enum
import struct


class SegmentMethod(Enum):
    """分段拼接方法"""
    FIXED = "fixed"                      
    FIXED_WINDOWED = "fixed_windowed"   
    RANDOM = "random"                    
    RANDOM_WINDOWED = "random_windowed"  
    RANDOM_PARTIAL = "random_partial"    


SEGMENT_METHOD_NAMES = {
    SegmentMethod.FIXED: "固定长度",
    SegmentMethod.FIXED_WINDOWED: "固定长度+窗函数",
    SegmentMethod.RANDOM: "随机长度",
    SegmentMethod.RANDOM_WINDOWED: "随机长度+窗函数",
    SegmentMethod.RANDOM_PARTIAL: "随机长度+部分窗(默认)",
}


def nature_field_amplitude(freq: float) -> float:
    """
    自然场幅度模型
    
    单位A/m，基于论文中的经验公式
    """
    x = np.log10(np.clip(freq, 1e-4, 1e4))
    
    result = (1.846 * np.sin(0.3578 * x + (-2.302)) +
              0.4855 * np.sin(1.466 * x + (-0.6232)) +
              0.07102 * np.sin(2.435 * x + (-2.001)) +
              0.05518 * np.sin(3.765 * x + (-0.8136)))
    
    return 10**result / (400 * np.pi) / 10


def single_freq_signal(amp: float, phase: float, freq: float,
                      sample_rate: float, n: int) -> np.ndarray:
    """
    生成单频时间序列: E(t) = A·cos(2πft + φ)
    """
    t = np.arange(n, dtype=np.float64) / sample_rate
    return amp * np.cos(2 * np.pi * freq * t + phase)


def hanning(n: int) -> np.ndarray:
    """Hanning窗"""
    return np.hanning(n)


def inv_hanning(n: int) -> np.ndarray:
    """逆Hanning窗"""
    return 1 - np.hanning(n)


@dataclass
class EMFields:
    """电磁场数据结构 (单频率点)"""
    freq: float
    ex1: complex; ey1: complex; hx1: complex; hy1: complex; hz1: complex
    ex2: complex; ey2: complex; hx2: complex; hy2: complex; hz2: complex


@dataclass
class Site:
    """测点数据结构"""
    name: str
    x: float
    y: float
    fields: List[EMFields]
    
    def frequencies(self) -> np.ndarray:
        return np.array([f.freq for f in self.fields])


class TimeSeriesGenerator:
    """
    时间序列生成器
    
    核心算法:
    1. 频域→时域转换: E(t) = A·cos(2πft + φ)
    2. 两正交源随机线性组合模拟自然源偏振变化
    3. 分段拼接模拟时变特性
    
    Example:
        >>> gen = TimeSeriesGenerator(2400, 1, 1000)
        >>> ex, ey, hx, hy, hz = gen.generate(3600, site)
    """
    
    def __init__(self, sample_rate: float, freq_min: float, freq_max: float,
                 segment_periods: float = 8.0, source_scale: float = 1.0,
                 method: SegmentMethod = SegmentMethod.RANDOM_PARTIAL):
        self.sample_rate = sample_rate
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.segment_periods = segment_periods
        self.source_scale = source_scale
        self.method = method
    
    def generate(self, duration: float, site: Site,
                 seed: Optional[int] = None) -> Tuple[np.ndarray, ...]:
        """
        生成合成时间序列
        
        Parameters:
            duration: 持续时间 (秒)
            site: 测点数据
            seed: 随机种子
        
        Returns:
            (ex, ey, hx, hy, hz) 五通道时间序列
        """
        n = int(duration * self.sample_rate)
        rng = np.random.default_rng(seed)
        
        ex = np.zeros(n)
        ey = np.zeros(n)
        hx = np.zeros(n)
        hy = np.zeros(n)
        hz = np.zeros(n)
        
        for f in site.fields:
            if not (self.freq_min <= f.freq <= self.freq_max):
                continue
            
            amps1 = np.array([abs(f.ex1), abs(f.ey1), abs(f.hx1), abs(f.hy1), abs(f.hz1)])
            phas1 = np.array([np.angle(f.ex1), np.angle(f.ey1), np.angle(f.hx1), 
                            np.angle(f.hy1), np.angle(f.hz1)])
            amps2 = np.array([abs(f.ex2), abs(f.ey2), abs(f.hx2), abs(f.hy2), abs(f.hz2)])
            phas2 = np.array([np.angle(f.ex2), np.angle(f.ey2), np.angle(f.hx2),
                            np.angle(f.hy2), np.angle(f.hz2)])
            
            ts1 = self._generate(amps1, phas1, f.freq, n, rng)
            ts2 = self._generate(amps2, phas2, f.freq, n, rng)
            
            for i, ch in enumerate([ex, ey, hx, hy, hz]):
                ch += ts1[i] + ts2[i]
        
        return ex, ey, hx, hy, hz
    
    def _generate(self, amps: np.ndarray, phas: np.ndarray,
                  freq: float, n: int, rng) -> List[np.ndarray]:
        """根据分段方法生成时间序列"""
        if self.method == SegmentMethod.FIXED:
            return self._fixed(amps, phas, freq, n, rng)
        elif self.method == SegmentMethod.FIXED_WINDOWED:
            return self._fixed_windowed(amps, phas, freq, n, rng)
        elif self.method == SegmentMethod.RANDOM:
            return self._random(amps, phas, freq, n, rng)
        elif self.method == SegmentMethod.RANDOM_WINDOWED:
            return self._random_windowed(amps, phas, freq, n, rng)
        else:
            return self._random_partial(amps, phas, freq, n, rng)
    
    def _fixed(self, amps, phas, freq, n, rng):
        """固定长度分段"""
        per = max(int(self.segment_periods / freq * self.sample_rate), 10)
        result = [np.zeros(n) for _ in range(5)]
        
        k = 0
        while k * per < n:
            da = rng.random() * 2
            dp = rng.random() * 2 * np.pi
            
            for i in range(5):
                ts = single_freq_signal(amps[i] * da * self.source_scale,
                                      phas[i] + np.pi + dp, freq,
                                      self.sample_rate, per)
                result[i][k*per:(k+1)*per] = ts
            k += 1
        
        return result
    
    def _fixed_windowed(self, amps, phas, freq, n, rng):
        """固定长度+窗函数"""
        per = max(int(self.segment_periods / freq * self.sample_rate), 10)
        window = hanning(per)
        result = [np.zeros(n) for _ in range(5)]
        
        k = 0
        while k * per < n:
            da = rng.random() * 2
            dp = rng.random() * 2 * np.pi
            
            for i in range(5):
                ts = single_freq_signal(amps[i] * da * self.source_scale,
                                      phas[i] + np.pi + dp, freq,
                                      self.sample_rate, per)
                result[i][k*per:(k+1)*per] = ts * window
            k += 1
        
        return result
    
    def _random(self, amps, phas, freq, n, rng):
        """随机长度分段"""
        result = [np.zeros(n) for _ in range(5)]
        left = n
        
        while left > 0:
            mean_len = int(self.segment_periods / freq * self.sample_rate)
            seg_len = max(int(abs(rng.normal(mean_len, mean_len/2))), 10)
            seg_len = min(seg_len, left)
            
            da = rng.random() * 2
            dp = rng.random() * 2 * np.pi
            start = n - left
            
            for i in range(5):
                ts = single_freq_signal(amps[i] * da * self.source_scale,
                                      phas[i] + np.pi + dp, freq,
                                      self.sample_rate, seg_len)
                result[i][start:start+seg_len] = ts
            
            left -= seg_len
        
        return result
    
    def _random_windowed(self, amps, phas, freq, n, rng):
        """随机长度+窗函数"""
        result = [np.zeros(n) for _ in range(5)]
        left = n
        
        while left > 0:
            mean_len = int(self.segment_periods / freq * self.sample_rate)
            seg_len = max(int(abs(rng.normal(mean_len, mean_len/2))), 10)
            seg_len = min(seg_len, left)
            
            window = hanning(seg_len)
            da = rng.random() * 2
            dp = rng.random() * 2 * np.pi
            start = n - left
            
            for i in range(5):
                ts = single_freq_signal(amps[i] * da * self.source_scale,
                                      phas[i] + np.pi + dp, freq,
                                      self.sample_rate, seg_len)
                result[i][start:start+seg_len] = ts * window
            
            left -= seg_len
        
        return result
    
    def _random_partial(self, amps, phas, freq, n, rng):
        """
        随机长度+部分窗 (默认方法)
        
        只在拼接处使用窗函数平滑，中间部分保持不变
        """
        result = [np.zeros(n) for _ in range(5)]
        wlen = max(int(self.sample_rate / 2 / freq) * 2, 2)
        window = inv_hanning(wlen)
        left = n
        
        while left > 0:
            mean_len = int(self.segment_periods / freq * self.sample_rate)
            seg_len = max(int(abs(rng.normal(mean_len, mean_len/2))), 10)
            seg_len = min(seg_len, left)
            
            da = rng.random() * 2
            dp = rng.random() * 2 * np.pi
            start = n - left
            
            for i in range(5):
                ts = single_freq_signal(amps[i] * da * self.source_scale,
                                      phas[i] + np.pi + dp, freq,
                                      self.sample_rate, seg_len)
                
                if seg_len > wlen * 2:
                    end = start + seg_len
                    result[i][start:start+wlen] += ts[:wlen] * window
                    result[i][end-wlen:end] += ts[-wlen:] * window
                    result[i][start+wlen:end-wlen] += ts[wlen:-wlen]
                else:
                    result[i][start:start+seg_len] += ts
            
            left -= seg_len
        
        return result


class MTSchema:
    """MT采集系统配置"""
    TS3 = {'name': 'TS3', 'rate': 2400, 'freq_min': 1, 'freq_max': 1000}
    TS4 = {'name': 'TS4', 'rate': 150, 'freq_min': 0.1, 'freq_max': 10}
    TS5 = {'name': 'TS5', 'rate': 15, 'freq_min': 1e-6, 'freq_max': 1}
    
    @classmethod
    def create(cls, name: str, **kwargs) -> TimeSeriesGenerator:
        """从配置名创建生成器"""
        config = getattr(cls, name.upper(), cls.TS3)
        return TimeSeriesGenerator(
            sample_rate=config['rate'],
            freq_min=config['freq_min'],
            freq_max=config['freq_max'],
            **kwargs
        )


def load_modem_file(filepath: str) -> List[Site]:
    """
    加载ModEM正演结果文件
    
    返回Site列表，每个Site包含多个频率的EMFields
    """
    sites = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('#') or not line:
            i += 1
            continue
        
        if '> Full_Impedance' in line:
            # 跳过标题行
            for _ in range(5):
                i += 1
                lines[i].strip()
            
            header = lines[i].strip().split()
            fcount = int(header[1])
            scount = int(header[2])
            
            i += 1
            
            # 读取每个测点
            for s in range(scount):
                fields = []
                for k in range(fcount):
                    # 读取Zxx, Zxy, Zyx, Zyy
                    for _ in range(4):
                        parts = lines[i].strip().split()
                        i += 1
                    
                    # 读取EM_Fields
                    em = EMFields(
                        freq=0,  # 待填充
                        ex1=complex(0,0), ey1=complex(0,0), hx1=complex(0,0),
                        hy1=complex(0,0), hz1=complex(0,0),
                        ex2=complex(0,0), ey2=complex(0,0), hx2=complex(0,0),
                        hy2=complex(0,0), hz2=complex(0,0)
                    )
                    fields.append(em)
                
                site = Site(name=f'Site{s}', x=0, y=0, fields=fields)
                sites.append(site)
        else:
            i += 1
    
    return sites
