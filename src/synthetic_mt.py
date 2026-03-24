"""
SyntheticMT - 大地电磁合成时间序列

基于论文和Delphi源码完整移植
Wang P, Chen X, Zhang Y (2023) Front. Earth Sci. 11:1086749
"""

import numpy as np
import struct
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
from datetime import datetime


class SyntheticMethod(Enum):
    """合成方法"""
    FIX = 0                         # 不分段
    FIXED_AVG = 1                   # 固定长度平均分段
    FIXED_AVG_WINDOWED = 2         # 固定长度平均分段+窗函数
    RANDOM_SEG = 3                  # 随机长度分段
    RANDOM_SEG_WINDOWED = 4         # 随机长度分段+窗函数
    RANDOM_SEG_PARTIAL = 5          # 随机长度分段+部分窗 (默认)


SYNTHETIC_METHOD_NAMES = {
    SyntheticMethod.FIX: 'No Segment',
    SyntheticMethod.FIXED_AVG: 'Fixed Length Average',
    SyntheticMethod.FIXED_AVG_WINDOWED: 'Fixed Length Average & Windowed',
    SyntheticMethod.RANDOM_SEG: 'Random Segment Length',
    SyntheticMethod.RANDOM_SEG_WINDOWED: 'Random Segment Length & Windowed',
    SyntheticMethod.RANDOM_SEG_PARTIAL: 'Random Segment Length & Partially Windowed',
}


class NoiseType(Enum):
    """噪声类型"""
    SQUARE_WAVE = 'square'      # 方波噪声
    TRIANGULAR = 'triangular'   # 三角波噪声
    IMPULSIVE = 'impulsive'    # 冲击噪声（随机尖峰）
    GAUSSIAN = 'gaussian'       # 高斯噪声
    POWERLINE = 'powerline'    # 工频干扰 (50/60Hz)


@dataclass
class NoiseConfig:
    """噪声配置"""
    noise_type: NoiseType = NoiseType.GAUSSIAN
    amplitude: float = 0.0      # 噪声幅度
    frequency: float = 0.0    # 噪声频率 (Hz), 用于周期性噪声
    probability: float = 0.01  # 冲击噪声概率
    phase: float = 0.0         # 相位 (rad)


TS_CONFIGS = {
    'TS2': {'sample_rate': 2400, 'freq_min': 10, 'freq_max': 12000},
    'TS3': {'sample_rate': 2400, 'freq_min': 1, 'freq_max': 1000},
    'TS4': {'sample_rate': 150, 'freq_min': 0.1, 'freq_max': 10},
    'TS5': {'sample_rate': 15, 'freq_min': 1e-6, 'freq_max': 1},
}


@dataclass
class EMFields:
    """单频率点的电磁场数据"""
    freq: float
    ex1: complex = complex(0, 0)
    ey1: complex = complex(0, 0)
    hx1: complex = complex(0, 0)
    hy1: complex = complex(0, 0)
    hz1: complex = complex(0, 0)
    ex2: complex = complex(0, 0)
    ey2: complex = complex(0, 0)
    hx2: complex = complex(0, 0)
    hy2: complex = complex(0, 0)
    hz2: complex = complex(0, 0)
    zxx: complex = complex(0, 0)
    zxy: complex = complex(0, 0)
    zyx: complex = complex(0, 0)
    zyy: complex = complex(0, 0)
    tzx: complex = complex(0, 0)
    tzy: complex = complex(0, 0)


@dataclass
class ForwardSite:
    """正演测点"""
    name: str
    x: float = 0.0
    y: float = 0.0
    fields: List[EMFields] = field(default_factory=list)
    
    def frequencies(self) -> np.ndarray:
        return np.array([f.freq for f in self.fields])
    
    def add_fields(self, f: EMFields):
        self.fields.append(f)
    
    def update_nature_magnetic_amplitude(self, scale_e: np.ndarray, scale_b: np.ndarray):
        """
        根据自然磁场强度更新电磁场幅值
        
        Parameters:
            scale_e: 电场缩放因子 (基于Hz幅值)
            scale_b: 磁场缩放因子 (基于Hz幅值)
        """
        for i, f in enumerate(self.fields):
            if i < len(scale_e):
                f.ex1 = complex(abs(f.ex1) * scale_e[i], f.ex1.imag)
                f.ey1 = complex(abs(f.ey1) * scale_e[i], f.ey1.imag)
                f.hx1 = complex(abs(f.hx1) * scale_b[i], f.hx1.imag)
                f.hy1 = complex(abs(f.hy1) * scale_b[i], f.hy1.imag)
                f.hz1 = complex(abs(f.hz1) * scale_b[i], f.hz1.imag)
                f.ex2 = complex(abs(f.ex2) * scale_e[i], f.ex2.imag)
                f.ey2 = complex(abs(f.ey2) * scale_e[i], f.ey2.imag)
                f.hx2 = complex(abs(f.hx2) * scale_b[i], f.hx2.imag)
                f.hy2 = complex(abs(f.hy2) * scale_b[i], f.hy2.imag)
                f.hz2 = complex(abs(f.hz2) * scale_b[i], f.hz2.imag)

    def interpolation(self, per_count: int):
        """
        Add interpolated frequency points between existing frequencies.
        
        Uses log-linear interpolation in frequency domain. For each pair of
        adjacent frequency points, adds per_count interpolated points.
        
        Parameters:
            per_count: Number of interpolated points to add between each pair
        """
        if len(self.fields) < 2:
            return
        
        # Sort fields by frequency first
        self.fields.sort(key=lambda f: f.freq)
        
        new_fields = []
        field_names = ['ex1', 'ey1', 'hx1', 'hy1', 'hz1', 'ex2', 'ey2', 
                       'hx2', 'hy2', 'hz2', 'zxx', 'zxy', 'zyx', 'zyy', 'tzx', 'tzy']
        
        for i in range(len(self.fields) - 1):
            f1 = self.fields[i]
            f2 = self.fields[i + 1]
            
            # Add the first field
            new_fields.append(f1)
            
            # Create interpolated points
            log_f1 = np.log10(f1.freq)
            log_f2 = np.log10(f2.freq)
            
            for j in range(1, per_count + 1):
                scale = j / (per_count + 1)
                new_freq = 10 ** (log_f1 + scale * (log_f2 - log_f1))
                
                new_field = EMFields(freq=new_freq)
                
                for name in field_names:
                    val1 = getattr(f1, name)
                    val2 = getattr(f2, name)
                    # Log-linear interpolation for complex values
                    new_val = complex(
                        10 ** (np.log10(abs(val1) + 1e-20) + scale * (np.log10(abs(val2) + 1e-20) - np.log10(abs(val1) + 1e-20))),
                        val1.imag + scale * (val2.imag - val1.imag)
                    )
                    setattr(new_field, name, new_val)
                
                new_fields.append(new_field)
        
        # Add the last field
        new_fields.append(self.fields[-1])
        
        self.fields = new_fields
    
    def negative_harmonic_factor(self):
        """
        Apply conjugate transformation to all field components.
        
        Applies .conjugate to all complex field components: ex1, ey1, hx1,
        hy1, hz1, ex2, ey2, hx2, hy2, hz2, zxx, zxy, zyx, zyy, tzx, tzy
        """
        field_names = ['ex1', 'ey1', 'hx1', 'hy1', 'hz1', 'ex2', 'ey2',
                       'hx2', 'hy2', 'hz2', 'zxx', 'zxy', 'zyx', 'zyy', 'tzx', 'tzy']
        
        for f in self.fields:
            for name in field_names:
                val = getattr(f, name)
                if isinstance(val, complex):
                    setattr(f, name, val.conjugate())
    
    def get_feh1(self):
        """
        Get fields with unit conversion (like Delphi GetFEH1).
        
        Returns:
            Tuple of (freq, Ex1, Ey1, Hx1, Hy1, Hz1, Ex2, Ey2, Hx2, Hy2, Hz2) arrays
            
        Unit conversions applied:
            - Ex *= 1E6
            - Ey *= 1E6
            - Hx *= 4E2 * np.pi
            - Hy *= 4E2 * np.pi
            - Hz *= 4E2 * np.pi
        """
        n = len(self.fields)
        fre = np.zeros(n)
        Ex1 = np.zeros(n, dtype=complex)
        Ey1 = np.zeros(n, dtype=complex)
        Hx1 = np.zeros(n, dtype=complex)
        Hy1 = np.zeros(n, dtype=complex)
        Hz1 = np.zeros(n, dtype=complex)
        Ex2 = np.zeros(n, dtype=complex)
        Ey2 = np.zeros(n, dtype=complex)
        Hx2 = np.zeros(n, dtype=complex)
        Hy2 = np.zeros(n, dtype=complex)
        Hz2 = np.zeros(n, dtype=complex)
        
        for i, f in enumerate(self.fields):
            fre[i] = f.freq
            Ex1[i] = f.ex1 * 1e6
            Ey1[i] = f.ey1 * 1e6
            Hx1[i] = f.hx1 * 4e2 * np.pi
            Hy1[i] = f.hy1 * 4e2 * np.pi
            Hz1[i] = f.hz1 * 4e2 * np.pi
            Ex2[i] = f.ex2 * 1e6
            Ey2[i] = f.ey2 * 1e6
            Hx2[i] = f.hx2 * 4e2 * np.pi
            Hy2[i] = f.hy2 * 4e2 * np.pi
            Hz2[i] = f.hz2 * 4e2 * np.pi
        
        return fre, Ex1, Ey1, Hx1, Hy1, Hz1, Ex2, Ey2, Hx2, Hy2, Hz2
    
    def add_calibration(self, responds: np.ndarray):
        """
        Apply system response calibration to fields.
        
        Parameters:
            responds: System response array of shape (5, n_frequencies)
                     Channel mapping: 0=Ex, 1=Ey, 2=Hx, 3=Hy, 4=Hz
                     
        Note:
            For electric fields (Ex, Ey): field /= responds[channel]
            For magnetic fields (Hx, Hy, Hz): field /= responds[channel] * 4E-7 * np.pi
        """
        if len(self.fields) == 0:
            return
        
        n_frequencies = len(self.fields)
        if responds.shape[1] != n_frequencies:
            raise ValueError(f"responds shape {responds.shape} doesn't match {n_frequencies} frequencies")
        
        field_mapping = {
            'ex1': 0, 'ey1': 1, 'hx1': 2, 'hy1': 3, 'hz1': 4,
            'ex2': 0, 'ey2': 1, 'hx2': 2, 'hy2': 3, 'hz2': 4,
        }
        
        magnetic_fields = {'hx1', 'hy1', 'hz1', 'hx2', 'hy2', 'hz2'}
        
        for i, f in enumerate(self.fields):
            for name, channel_idx in field_mapping.items():
                val = getattr(f, name)
                resp = responds[channel_idx, i]
                
                if abs(resp) > 1e-20:
                    if name in magnetic_fields:
                        val = val / (resp * 4e-7 * np.pi)
                    else:
                        val = val / resp
                    setattr(f, name, val)


def nature_magnetic_amplitude(freq: float) -> float:
    """
    计算自然磁场强度幅值 (单位: nT)
    
    基于MT频率范围和典型自然源磁场强度统计
    磁场强度随频率变化呈幂律分布
    
    Parameters:
        freq: 频率 (Hz)
    
    Returns:
        磁场幅值 (nT)
    
    Note:
        参考: Wang et al. (2023) 论文中的自然磁场模型
        使用分段幂律近似典型MT源的磁场强度谱
    """
    period = 1.0 / freq if freq > 0 else float('inf')
    
    if freq <= 0:
        return 0.0
    
    if freq < 1e-5:
        b = 100.0 * (freq / 1e-5) ** 0.3
    elif freq < 1e-3:
        b = 100.0 * (freq / 1e-5) ** 0.3
    elif freq < 0.01:
        b = 30.0 * (freq / 1e-3) ** 0.2
    elif freq < 0.1:
        b = 20.0 * (freq / 0.01) ** 0.1
    elif freq < 1.0:
        b = 10.0 * (freq / 0.1) ** 0.05
    elif freq < 10.0:
        b = 5.0 * (freq / 1.0) ** -0.1
    elif freq < 100.0:
        b = 3.0 * (freq / 10.0) ** -0.2
    elif freq < 1000.0:
        b = 1.0 * (freq / 100.0) ** -0.3
    else:
        b = 0.3 * (freq / 1000.0) ** -0.4
    
    return max(b, 0.001)


def calculate_mt_scale_factors(site: ForwardSite) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算MT测量的缩放因子
    
    基于自然磁场强度和测点响应计算缩放因子
    用于将正演结果缩放到实际观测水平
    
    Parameters:
        site: 正演测点
    
    Returns:
        (scale_e, scale_b): 电场和磁场缩放因子数组
    """
    freqs = site.frequencies()
    n = len(freqs)
    
    scale_e = np.ones(n)
    scale_b = np.ones(n)
    
    for i, f in enumerate(freqs):
        nat_b = nature_magnetic_amplitude(f)
        
        if i < len(site.fields):
            hx1_mag = abs(site.fields[i].hx1)
            hy1_mag = abs(site.fields[i].hy1)
            hx2_mag = abs(site.fields[i].hx2)
            hy2_mag = abs(site.fields[i].hy2)
            
            h1_mag = np.sqrt(hx1_mag**2 + hy1_mag**2)
            h2_mag = np.sqrt(hx2_mag**2 + hy2_mag**2)
            
            if h1_mag > 0:
                scale_b[i] = nat_b / h1_mag
            if h2_mag > 0:
                scale_b[i] = min(scale_b[i], nat_b / h2_mag)
            
            e1_mag = np.sqrt(abs(site.fields[i].ex1)**2 + abs(site.fields[i].ey1)**2)
            e2_mag = np.sqrt(abs(site.fields[i].ex2)**2 + abs(site.fields[i].ey2)**2)
            
            if e1_mag > 0 or e2_mag > 0:
                scale_e[i] = min(
                    (nat_b * 377.0) / e1_mag if e1_mag > 0 else float('inf'),
                    (nat_b * 377.0) / e2_mag if e2_mag > 0 else float('inf')
                )
    
    return scale_e, scale_b


def freq_to_time(amp: float, phase: float, freq: float, 
                 sample_rate: float, n: int, output: np.ndarray) -> None:
    """频域转时域: E(t) = A * cos(2πft + φ)"""
    t = np.arange(n, dtype=np.float64) / sample_rate
    output[:] = amp * np.cos(2 * np.pi * freq * t + phase)


def hanning_window(n: int) -> np.ndarray:
    """Hanning窗"""
    return np.hanning(n)


def inv_hanning_window(n: int) -> np.ndarray:
    """逆Hanning窗"""
    return 1 - np.hanning(n)


class SyntheticSchema:
    """合成参数配置"""
    
    def __init__(self, name: str = 'TS3', sample_rate: float = 2400,
                 freq_min: float = 1, freq_max: float = 1000,
                 synthetic_periods: float = 8.0,
                 source_scale: float = 1.0,
                 continuous: bool = False):
        self.name = name
        self.sample_rate = sample_rate
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.synthetic_periods = synthetic_periods
        self.source_scale = source_scale
        self.continuous = continuous
    
    @classmethod
    def from_ts(cls, ts_name: str, **kwargs):
        config = TS_CONFIGS.get(ts_name, TS_CONFIGS['TS3'])
        config = config.copy()
        config.update(kwargs)
        return cls(name=ts_name, **config)


def load_modem_file(filepath: str) -> List[ForwardSite]:
    """
    加载ModEM正演结果文件
    
    ModEM格式:
    - > Full_Impedance: 阻抗数据
    - > Full_Vertical_Components: Tipper数据
    - > EM_Fields: 电磁场数据
    
    单位转换:
    - [mv/Km]/[nT] -> [V/m]/[A/m]: 除以(1E6/4E2/π)
    - Ex := val * (4E2 * π) [mv/Km] -> [V/m]
    - Hx := val * 1E9 [nT] -> [A/m]
    """
    with open(filepath, 'r', errors='ignore') as f:
        content = f.read()
    
    sites = []
    blocks = content.split('>')
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        lines = block.split('\n')
        header = lines[0].strip()
        
        if 'Full_Impedance' in header:
            sites = _parse_impedance_block(lines[1:])
        elif 'Full_Vertical_Components' in header:
            sites = _parse_tipper_block(lines[1:], sites)
        elif 'EM_Fields' in header:
            sites = _parse_emfields_block(lines[1:], sites)
    
    for site in sites:
        site.fields.sort(key=lambda x: x.freq)
    
    return sites


def _parse_impedance_block(lines: List[str]) -> List[ForwardSite]:
    """解析阻抗数据块"""
    sites = []
    idx = 0
    
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    
    if idx >= len(lines):
        return sites
    
    header = lines[idx].strip().split()
    if len(header) < 3:
        return sites
    
    try:
        fcount = int(header[1])
        scount = int(header[2])
    except ValueError:
        return sites
    
    idx += 1
    
    for s in range(scount):
        if idx >= len(lines):
            break
        
        site = ForwardSite(name=f'Site{s}')
        
        site_fields = []
        for k in range(fcount):
            if idx >= len(lines):
                break
            
            parts = lines[idx].strip().split()
            if len(parts) >= 10:
                try:
                    freq = float(parts[0])
                    zxx = complex(float(parts[1]), float(parts[2]))
                    zxy = complex(float(parts[3]), float(parts[4]))
                    zyx = complex(float(parts[5]), float(parts[6]))
                    zyy = complex(float(parts[7]), float(parts[8]))
                    
                    scale = 1e6 / 4e2 / np.pi
                    fld = EMFields(
                        freq=freq,
                        zxx=zxx / scale,
                        zxy=zxy / scale,
                        zyx=zyx / scale,
                        zyy=zyy / scale,
                    )
                    site_fields.append(fld)
                except (ValueError, IndexError):
                    pass
            idx += 1
        
        site.fields = site_fields
        if site.fields:
            sites.append(site)
    
    return sites


def _parse_tipper_block(lines: List[str], sites: List[ForwardSite]) -> List[ForwardSite]:
    """解析Tipper数据块"""
    if not sites:
        return sites
    
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    
    if idx >= len(lines):
        return sites
    
    header = lines[idx].strip().split()
    if len(header) < 3:
        return sites
    
    try:
        fcount = int(header[1])
    except ValueError:
        return sites
    
    idx += 1
    
    for s, site in enumerate(sites):
        for k, fld in enumerate(site.fields):
            if idx >= len(lines):
                break
            
            parts = lines[idx].strip().split()
            if len(parts) >= 10:
                try:
                    tzx = complex(float(parts[1]), float(parts[2]))
                    tzy = complex(float(parts[3]), float(parts[4]))
                    site.fields[k].tzx = tzx
                    site.fields[k].tzy = tzy
                except (ValueError, IndexError):
                    pass
            idx += 1
    
    return sites


def _parse_emfields_block(lines: List[str], sites: List[ForwardSite]) -> List[ForwardSite]:
    """解析EM场数据块"""
    if not sites:
        return sites
    
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    
    if idx >= len(lines):
        return sites
    
    header = lines[idx].strip().split()
    if len(header) < 3:
        return sites
    
    try:
        fcount = int(header[1])
        n_sites = int(header[2])
    except ValueError:
        return sites
    
    idx += 1
    
    total_fields = fcount * n_sites
    
    for s, site in enumerate(sites):
        for k, fld in enumerate(site.fields):
            em_vals = []
            for _ in range(10):
                if idx >= len(lines):
                    break
                parts = lines[idx].strip().split()
                if len(parts) >= 10:
                    try:
                        em_vals.append((float(parts[8]), float(parts[9])))
                    except (ValueError, IndexError):
                        em_vals.append((0.0, 0.0))
                idx += 1
            
            if len(em_vals) >= 10:
                site.fields[k].ex1 = complex(*em_vals[0]) * (4e2 * np.pi)
                site.fields[k].ey1 = complex(*em_vals[1]) * (4e2 * np.pi)
                site.fields[k].hx1 = complex(*em_vals[2]) * 1e9
                site.fields[k].hy1 = complex(*em_vals[3]) * 1e9
                site.fields[k].hz1 = complex(*em_vals[4]) * 1e9
                site.fields[k].ex2 = complex(*em_vals[5]) * (4e2 * np.pi)
                site.fields[k].ey2 = complex(*em_vals[6]) * (4e2 * np.pi)
                site.fields[k].hx2 = complex(*em_vals[7]) * 1e9
                site.fields[k].hy2 = complex(*em_vals[8]) * 1e9
                site.fields[k].hz2 = complex(*em_vals[9]) * 1e9
    
    return sites


class SyntheticTimeSeries:
    """合成时间序列主类"""
    
    def __init__(self, schema: SyntheticSchema, 
                 method: SyntheticMethod = SyntheticMethod.RANDOM_SEG_PARTIAL):
        self.schema = schema
        self.method = method
    
    def generate(self, begin_time: datetime, end_time: datetime,
                 site: ForwardSite, seed: Optional[int] = None) -> Tuple[np.ndarray, ...]:
        """
        生成合成时间序列
        
        Parameters:
            begin_time: 开始时间
            end_time: 结束时间
            site: 正演测点数据
            seed: 随机种子
        
        Returns:
            (ex, ey, hx, hy, hz) 五通道时间序列
        """
        delta_t = (end_time - begin_time).total_seconds()
        n_samples = int(delta_t * self.schema.sample_rate)
        
        rng = np.random.default_rng(seed)
        
        ex = np.zeros(n_samples, dtype=np.float64)
        ey = np.zeros(n_samples, dtype=np.float64)
        hx = np.zeros(n_samples, dtype=np.float64)
        hy = np.zeros(n_samples, dtype=np.float64)
        hz = np.zeros(n_samples, dtype=np.float64)
        
        for f in site.fields:
            if not (self.schema.freq_min <= f.freq <= self.schema.freq_max):
                continue
            
            delta_pha1 = delta_t / (2.0 * np.pi)
            delta_pha1 = delta_pha1 - (int(delta_pha1 * 180 / np.pi) // 360) * 2 * np.pi
            delta_pha2 = delta_pha1 + np.pi / 4
            
            amps1 = np.array([
                abs(f.ex1), abs(f.ey1), abs(f.hx1), abs(f.hy1), abs(f.hz1)
            ]) * self.schema.source_scale
            phas1 = np.array([
                np.angle(f.ex1) + np.pi + delta_pha1,
                np.angle(f.ey1) + np.pi + delta_pha1,
                np.angle(f.hx1) + delta_pha1,
                np.angle(f.hy1) + delta_pha1,
                np.angle(f.hz1) + delta_pha1,
            ])
            
            amps2 = np.array([
                abs(f.ex2), abs(f.ey2), abs(f.hx2), abs(f.hy2), abs(f.hz2)
            ]) * self.schema.source_scale
            phas2 = np.array([
                np.angle(f.ex2) + np.pi + delta_pha2,
                np.angle(f.ey2) + np.pi + delta_pha2,
                np.angle(f.hx2) + delta_pha2,
                np.angle(f.hy2) + delta_pha2,
                np.angle(f.hz2) + delta_pha2,
            ])
            
            if self.method == SyntheticMethod.FIX:
                ts1, ts2 = self._fix_segment(amps1, phas1, amps2, phas2, f.freq, n_samples, rng)
            elif self.method == SyntheticMethod.FIXED_AVG:
                ts1, ts2 = self._fixed_avg_segment(amps1, phas1, amps2, phas2, f.freq, n_samples, rng)
            elif self.method == SyntheticMethod.FIXED_AVG_WINDOWED:
                ts1, ts2 = self._fixed_avg_windowed(amps1, phas1, amps2, phas2, f.freq, n_samples, rng)
            elif self.method == SyntheticMethod.RANDOM_SEG:
                ts1, ts2 = self._random_segment(amps1, phas1, amps2, phas2, f.freq, n_samples, rng)
            elif self.method == SyntheticMethod.RANDOM_SEG_WINDOWED:
                ts1, ts2 = self._random_windowed(amps1, phas1, amps2, phas2, f.freq, n_samples, rng)
            else:
                ts1, ts2 = self._random_partial(amps1, phas1, amps2, phas2, f.freq, n_samples, rng)
            
            ex += ts1[0] + ts2[0]
            ey += ts1[1] + ts2[1]
            hx += ts1[2] + ts2[2]
            hy += ts1[3] + ts2[3]
            hz += ts1[4] + ts2[4]
        
        return ex, ey, hx, hy, hz
    
    def _fix_segment(self, amps1, phas1, amps2, phas2, freq, n, rng):
        """不分段"""
        ex1 = np.zeros(n); ey1 = np.zeros(n); hx1 = np.zeros(n)
        hy1 = np.zeros(n); hz1 = np.zeros(n)
        ex2 = np.zeros(n); ey2 = np.zeros(n); hx2 = np.zeros(n)
        hy2 = np.zeros(n); hz2 = np.zeros(n)
        
        da1 = rng.random() * 2
        dp1 = rng.random() * 2 * np.pi
        da2 = rng.random() * 2
        dp2 = rng.random() * 2 * np.pi
        
        for i, (a1, p1, a2, p2) in enumerate(zip(amps1, phas1, amps2, phas2)):
            tmp = np.zeros(n)
            freq_to_time(a1 * da1, p1 + dp1, freq, self.schema.sample_rate, n, tmp)
            [ex1, ey1, hx1, hy1, hz1][i][:] = tmp
            freq_to_time(a2 * da2, p2 + dp2, freq, self.schema.sample_rate, n, tmp)
            [ex2, ey2, hx2, hy2, hz2][i][:] = tmp
        
        return (ex1, ey1, hx1, hy1, hz1), (ex2, ey2, hx2, hy2, hz2)
    
    def _fixed_avg_segment(self, amps1, phas1, amps2, phas2, freq, n, rng):
        """固定长度平均分段"""
        per_count = max(int(self.schema.sample_rate / freq * self.schema.synthetic_periods), 10)
        if per_count > n:
            per_count = n
        
        ex1 = np.zeros(n); ey1 = np.zeros(n); hx1 = np.zeros(n)
        hy1 = np.zeros(n); hz1 = np.zeros(n)
        ex2 = np.zeros(n); ey2 = np.zeros(n); hx2 = np.zeros(n)
        hy2 = np.zeros(n); hz2 = np.zeros(n)
        
        s_count = n // per_count
        
        for k in range(s_count):
            da1 = rng.random() * 2
            dp1 = rng.random() * 2 * np.pi
            da2 = rng.random() * 2
            dp2 = rng.random() * 2 * np.pi
            
            for i, (a1, p1, a2, p2) in enumerate(zip(amps1, phas1, amps2, phas2)):
                tmp = np.zeros(per_count)
                freq_to_time(a1 * da1, p1 + dp1, freq, self.schema.sample_rate, per_count, tmp)
                [ex1, ey1, hx1, hy1, hz1][i][k*per_count:(k+1)*per_count] = tmp
                freq_to_time(a2 * da2, p2 + dp2, freq, self.schema.sample_rate, per_count, tmp)
                [ex2, ey2, hx2, hy2, hz2][i][k*per_count:(k+1)*per_count] = tmp
        
        left = n - per_count * s_count
        if left > 0:
            k = s_count
            da1 = rng.random() * 2
            dp1 = rng.random() * 2 * np.pi
            da2 = rng.random() * 2
            dp2 = rng.random() * 2 * np.pi
            
            for i, (a1, p1, a2, p2) in enumerate(zip(amps1, phas1, amps2, phas2)):
                tmp = np.zeros(left)
                freq_to_time(a1 * da1, p1 + dp1, freq, self.schema.sample_rate, left, tmp)
                [ex1, ey1, hx1, hy1, hz1][i][k*per_count:] = tmp
                freq_to_time(a2 * da2, p2 + dp2, freq, self.schema.sample_rate, left, tmp)
                [ex2, ey2, hx2, hy2, hz2][i][k*per_count:] = tmp
        
        return (ex1, ey1, hx1, hy1, hz1), (ex2, ey2, hx2, hy2, hz2)
    
    def _fixed_avg_windowed(self, amps1, phas1, amps2, phas2, freq, n, rng):
        """固定长度平均分段+窗函数"""
        per_count = max(int(self.schema.sample_rate / freq * self.schema.synthetic_periods), 10)
        if per_count > n:
            per_count = n
        
        window = hanning_window(per_count)
        
        ex1 = np.zeros(n); ey1 = np.zeros(n); hx1 = np.zeros(n)
        hy1 = np.zeros(n); hz1 = np.zeros(n)
        ex2 = np.zeros(n); ey2 = np.zeros(n); hx2 = np.zeros(n)
        hy2 = np.zeros(n); hz2 = np.zeros(n)
        
        s_count = n // per_count
        da1 = dp1 = da2 = dp2 = 0.0
        
        for k in range(s_count):
            da1 = rng.random() * 2
            dp1 = rng.random() * 2 * np.pi
            da2 = rng.random() * 2
            dp2 = rng.random() * 2 * np.pi
            
            for i, (a1, p1, a2, p2) in enumerate(zip(amps1, phas1, amps2, phas2)):
                tmp = np.zeros(per_count)
                freq_to_time(a1 * da1, p1 + dp1, freq, self.schema.sample_rate, per_count, tmp)
                [ex1, ey1, hx1, hy1, hz1][i][k*per_count:(k+1)*per_count] = tmp * window
                freq_to_time(a2 * da2, p2 + dp2, freq, self.schema.sample_rate, per_count, tmp)
                [ex2, ey2, hx2, hy2, hz2][i][k*per_count:(k+1)*per_count] = tmp * window
        
        left = n - per_count * s_count
        if left > 0:
            k = s_count
            for i, (a1, p1, a2, p2) in enumerate(zip(amps1, phas1, amps2, phas2)):
                tmp = np.zeros(left)
                freq_to_time(a1 * da1, p1 + dp1, freq, self.schema.sample_rate, left, tmp)
                [ex1, ey1, hx1, hy1, hz1][i][k*per_count:] = tmp
                freq_to_time(a2 * da2, p2 + dp2, freq, self.schema.sample_rate, left, tmp)
                [ex2, ey2, hx2, hy2, hz2][i][k*per_count:] = tmp
        
        return (ex1, ey1, hx1, hy1, hz1), (ex2, ey2, hx2, hy2, hz2)
    
    def _random_segment(self, amps1, phas1, amps2, phas2, freq, n, rng):
        """随机长度分段"""
        ex1 = np.zeros(n); ey1 = np.zeros(n); hx1 = np.zeros(n)
        hy1 = np.zeros(n); hz1 = np.zeros(n)
        ex2 = np.zeros(n); ey2 = np.zeros(n); hx2 = np.zeros(n)
        hy2 = np.zeros(n); hz2 = np.zeros(n)
        
        left = n
        while left > 0:
            mean_len = int(self.schema.sample_rate / freq * self.schema.synthetic_periods)
            seg_len = int(abs(rng.normal(mean_len, mean_len / 2)))
            seg_len = max(seg_len, 10)
            seg_len = min(seg_len, left)
            
            da1 = rng.random() * 2
            dp1 = rng.random() * 2 * np.pi
            da2 = rng.random() * 2
            dp2 = rng.random() * 2 * np.pi
            
            start = n - left
            
            for i, (a1, p1, a2, p2) in enumerate(zip(amps1, phas1, amps2, phas2)):
                tmp = np.zeros(seg_len)
                freq_to_time(a1 * da1, p1 + dp1, freq, self.schema.sample_rate, seg_len, tmp)
                [ex1, ey1, hx1, hy1, hz1][i][start:start+seg_len] = tmp
                freq_to_time(a2 * da2, p2 + dp2, freq, self.schema.sample_rate, seg_len, tmp)
                [ex2, ey2, hx2, hy2, hz2][i][start:start+seg_len] = tmp
            
            left -= seg_len
        
        return (ex1, ey1, hx1, hy1, hz1), (ex2, ey2, hx2, hy2, hz2)
    
    def _random_windowed(self, amps1, phas1, amps2, phas2, freq, n, rng):
        """随机长度分段+窗函数"""
        ex1 = np.zeros(n); ey1 = np.zeros(n); hx1 = np.zeros(n)
        hy1 = np.zeros(n); hz1 = np.zeros(n)
        ex2 = np.zeros(n); ey2 = np.zeros(n); hx2 = np.zeros(n)
        hy2 = np.zeros(n); hz2 = np.zeros(n)
        
        left = n
        while left > 0:
            mean_len = int(self.schema.sample_rate / freq * self.schema.synthetic_periods)
            seg_len = int(abs(rng.normal(mean_len, mean_len / 2)))
            seg_len = max(seg_len, 10)
            seg_len = min(seg_len, left)
            
            window = hanning_window(seg_len)
            
            da1 = rng.random() * 2
            dp1 = rng.random() * 2 * np.pi
            da2 = rng.random() * 2
            dp2 = rng.random() * 2 * np.pi
            
            start = n - left
            
            for i, (a1, p1, a2, p2) in enumerate(zip(amps1, phas1, amps2, phas2)):
                tmp = np.zeros(seg_len)
                freq_to_time(a1 * da1, p1 + dp1, freq, self.schema.sample_rate, seg_len, tmp)
                [ex1, ey1, hx1, hy1, hz1][i][start:start+seg_len] = tmp * window
                freq_to_time(a2 * da2, p2 + dp2, freq, self.schema.sample_rate, seg_len, tmp)
                [ex2, ey2, hx2, hy2, hz2][i][start:start+seg_len] = tmp * window
            
            left -= seg_len
        
        return (ex1, ey1, hx1, hy1, hz1), (ex2, ey2, hx2, hy2, hz2)
    
    def _random_partial(self, amps1, phas1, amps2, phas2, freq, n, rng):
        """随机长度分段+部分窗 (默认方法)"""
        ex1 = np.zeros(n); ey1 = np.zeros(n); hx1 = np.zeros(n)
        hy1 = np.zeros(n); hz1 = np.zeros(n)
        ex2 = np.zeros(n); ey2 = np.zeros(n); hx2 = np.zeros(n)
        hy2 = np.zeros(n); hz2 = np.zeros(n)
        
        wlen = max(int(self.schema.sample_rate / 2 / freq) * 2, 2)
        window = inv_hanning_window(wlen)
        
        left = n
        while left > 0:
            mean_len = int(self.schema.sample_rate / freq * self.schema.synthetic_periods)
            seg_len = int(abs(rng.normal(mean_len, mean_len / 2)))
            seg_len = max(seg_len, 10)
            seg_len = min(seg_len, left)
            
            da1 = rng.random() * 2
            dp1 = rng.random() * 2 * np.pi
            da2 = rng.random() * 2
            dp2 = rng.random() * 2 * np.pi
            
            start = n - left
            
            for i, (a1, p1, a2, p2) in enumerate(zip(amps1, phas1, amps2, phas2)):
                tmp = np.zeros(seg_len)
                freq_to_time(a1 * da1, p1 + dp1, freq, self.schema.sample_rate, seg_len, tmp)
                
                if seg_len > wlen * 2:
                    end = start + seg_len
                    [ex1, ey1, hx1, hy1, hz1][i][start:start+wlen] += tmp[:wlen] * window
                    [ex1, ey1, hx1, hy1, hz1][i][end-wlen:end] += tmp[-wlen:] * window
                    [ex1, ey1, hx1, hy1, hz1][i][start+wlen:end-wlen] += tmp[wlen:-wlen]
                else:
                    [ex1, ey1, hx1, hy1, hz1][i][start:start+seg_len] += tmp
                
                freq_to_time(a2 * da2, p2 + dp2, freq, self.schema.sample_rate, seg_len, tmp)
                
                if seg_len > wlen * 2:
                    end = start + seg_len
                    [ex2, ey2, hx2, hy2, hz2][i][start:start+wlen] += tmp[:wlen] * window
                    [ex2, ey2, hx2, hy2, hz2][i][end-wlen:end] += tmp[-wlen:] * window
                    [ex2, ey2, hx2, hy2, hz2][i][start+wlen:end-wlen] += tmp[wlen:-wlen]
                else:
                    [ex2, ey2, hx2, hy2, hz2][i][start:start+seg_len] += tmp
            
            left -= seg_len
        
        return (ex1, ey1, hx1, hy1, hz1), (ex2, ey2, hx2, hy2, hz2)


def create_test_site() -> ForwardSite:
    """创建测试用正演数据"""
    fields = []
    freqs = np.logspace(-2, 3, 20)
    
    for f in freqs:
        fields.append(EMFields(
            freq=f,
            ex1=complex(1.0, 0.1), ey1=complex(0.8, -0.1),
            hx1=complex(0.01, 0), hy1=complex(0.01, 0), hz1=complex(0, 0),
            ex2=complex(0.9, -0.1), ey2=complex(-0.8, 0.1),
            hx2=complex(0.01, 0), hy2=complex(-0.01, 0), hz2=complex(0, 0),
        ))
    
    return ForwardSite(name='Test', x=0, y=0, fields=fields)


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


@dataclass
class CalibrationData:
    """标定数据"""
    channel: int
    serial_number: int
    sample_rate: int
    frequency: float
    amplitude_re: float
    amplitude_im: float
    phase_re: float
    phase_im: float


class ClbFile:
    """
    Phoenix CLB标定文件读写类
    
    CLB文件格式: 二进制格式, 包含通道标定响应参数
    """
    
    CAL_CHANNELS = {0: 'Ex', 1: 'Ey', 2: 'Hx', 3: 'Hy', 4: 'Hz'}
    
    def __init__(self, path: Optional[str] = None):
        self.calibrations: List[CalibrationData] = []
        if path:
            self.load(path)
    
    def load(self, path: str) -> None:
        """加载CLB标定文件"""
        with open(path, 'rb') as f:
            data = f.read()
        
        self.calibrations = []
        offset = 0
        
        while offset < len(data) - 64:
            try:
                channel = data[offset]
                serial = int.from_bytes(data[offset+1:offset+3], 'little')
                sample_rate = int.from_bytes(data[offset+3:offset+5], 'little')
                frequency = struct.unpack('f', data[offset+5:offset+9])[0]
                amp_re = struct.unpack('f', data[offset+9:offset+13])[0]
                amp_im = struct.unpack('f', data[offset+13:offset+17])[0]
                phase_re = struct.unpack('f', data[offset+17:offset+21])[0]
                phase_im = struct.unpack('f', data[offset+21:offset+25])[0]
                
                cal = CalibrationData(
                    channel=channel,
                    serial_number=serial,
                    sample_rate=sample_rate,
                    frequency=frequency,
                    amplitude_re=amp_re,
                    amplitude_im=amp_im,
                    phase_re=phase_re,
                    phase_im=phase_im,
                )
                self.calibrations.append(cal)
                offset += 64
            except Exception:
                break
    
    def save(self, path: str) -> None:
        """保存CLB标定文件"""
        with open(path, 'wb') as f:
            for cal in self.calibrations:
                f.write(bytes([cal.channel]))
                f.write(cal.serial_number.to_bytes(2, 'little'))
                f.write(cal.sample_rate.to_bytes(2, 'little'))
                f.write(struct.pack('f', cal.frequency))
                f.write(struct.pack('f', cal.amplitude_re))
                f.write(struct.pack('f', cal.amplitude_im))
                f.write(struct.pack('f', cal.phase_re))
                f.write(struct.pack('f', cal.phase_im))
                f.write(b'\x00' * 39)
    
    def get_calibration(self, channel: int, frequency: float) -> Optional[CalibrationData]:
        """获取指定通道和频率的标定数据"""
        best_match = None
        best_diff = float('inf')
        
        for cal in self.calibrations:
            if cal.channel == channel:
                diff = abs(cal.frequency - frequency)
                if diff < best_diff:
                    best_diff = diff
                    best_match = cal
        
        return best_match
    
    def apply_calibration(self, data: np.ndarray, channel: int,
                         sample_rate: float) -> np.ndarray:
        """
        应用标定校正到时间序列
        
        Parameters:
            data: 时间序列数据
            channel: 通道号 (0-4)
            sample_rate: 采样率
        
        Returns:
            校正后的时间序列
        """
        freqs = np.fft.rfftfreq(len(data), 1.0/sample_rate)
        fft_data = np.fft.rfft(data)
        
        for i, f in enumerate(freqs):
            if f == 0:
                continue
            cal = self.get_calibration(channel, f)
            if cal:
                resp = complex(cal.amplitude_re, cal.amplitude_im) * \
                       np.exp(1j * complex(cal.phase_re, cal.phase_im))
                if abs(resp) > 1e-10:
                    fft_data[i] /= resp
        
        return np.fft.irfft(fft_data, n=len(data))


class ClcFile:
    """
    Phoenix CLC标定配置文件读写类
    
    CLC文件是ASCII格式的标定配置文件
    """
    
    def __init__(self, path: Optional[str] = None):
        self.info = {}
        if path:
            self.load(path)
    
    def load(self, path: str) -> None:
        """加载CLC标定配置文件"""
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    key = parts[0]
                    try:
                        if parts[1] == 'd':
                            self.info[key] = float(parts[2])
                        elif parts[1] == 'i':
                            self.info[key] = int(parts[2])
                        else:
                            self.info[key] = ' '.join(parts[2:])
                    except (ValueError, IndexError):
                        pass
    
    def save(self, path: str) -> None:
        """保存CLC标定配置文件"""
        with open(path, 'w') as f:
            f.write('# Phoenix CLC Calibration File\n\n')
            for key, value in self.info.items():
                if isinstance(value, float):
                    f.write(f'{key} d {value}\n')
                elif isinstance(value, int):
                    f.write(f'{key} i {value}\n')
                else:
                    f.write(f'{key} s {value}\n')
    
    def __getitem__(self, key: str):
        return self.info.get(key)
    
    def __setitem__(self, key: str, value):
        self.info[key] = value


def save_gmt_timeseries(dir_path: str, site_name: str, 
                        ex: np.ndarray, ey: np.ndarray,
                        hx: np.ndarray, hy: np.ndarray, hz: np.ndarray,
                        begin_time: datetime, sample_rate: float,
                        unit: str = 'V/m') -> str:
    """
    保存GMT格式时间序列文本文件
    
    GMT (Generic Mapping Tools) 兼容的时间序列格式
    每行格式: Year Month Day Hour Minute Second Ex Ey Hx Hy Hz
    
    Parameters:
        dir_path: 输出目录
        site_name: 测点名称
        ex, ey, hx, hy, hz: 电磁场时间序列
        begin_time: 开始时间
        sample_rate: 采样率 (Hz)
        unit: 数据单位 ('V/m' for E, 'A/m' for H)
    
    Returns:
        保存的文件路径
    """
    from pathlib import Path
    
    output_dir = Path(dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / f"{site_name}.txt"
    
    n_samples = len(ex)
    dt = 1.0 / sample_rate
    
    with open(file_path, 'w') as f:
        f.write("# GMT时间序列数据\n")
        f.write(f"# 测点: {site_name}\n")
        f.write(f"# 开始时间: {begin_time.isoformat()}\n")
        f.write(f"# 采样率: {sample_rate} Hz\n")
        f.write(f"# 单位: Ex Ey [V/m], Hx Hy Hz [A/m]\n")
        f.write("# Year Month Day Hour Minute Second Ex Ey Hx Hy Hz\n")
        
        current_time = begin_time
        
        for i in range(n_samples):
            if i > 0 and i % 100000 == 0:
                current_time = begin_time.replace(
                    second=begin_time.second + int(i / sample_rate)
                )
            
            f.write(
                f"{current_time.year:4d} "
                f"{current_time.month:02d} "
                f"{current_time.day:02d} "
                f"{current_time.hour:02d} "
                f"{current_time.minute:02d} "
                f"{current_time.second:02d} "
                f"{ex[i]:15.6e} "
                f"{ey[i]:15.6e} "
                f"{hx[i]:15.6e} "
                f"{hy[i]:15.6e} "
                f"{hz[i]:15.6e}\n"
            )
            
            current_time = datetime.fromtimestamp(
                begin_time.timestamp() + i * dt
            )
    
    return str(file_path)


def save_csv_timeseries(file_path: str,
                        ex: np.ndarray, ey: np.ndarray,
                        hx: np.ndarray, hy: np.ndarray, hz: np.ndarray,
                        header: Optional[str] = None) -> str:
    """
    保存CSV格式时间序列
    
    Parameters:
        file_path: 输出文件路径
        ex, ey, hx, hy, hz: 电磁场时间序列
        header: CSV头注释
    
    Returns:
        保存的文件路径
    """
    data = np.column_stack([ex, ey, hx, hy, hz])
    
    if header is None:
        header = "Ex(V/m),Ey(V/m),Hx(A/m),Hy(A/m),Hz(A/m)"
    
    np.savetxt(file_path, data, delimiter=',', header=header)
    
    return file_path


def save_numpy_timeseries(file_path: str,
                          ex: np.ndarray, ey: np.ndarray,
                          hx: np.ndarray, hy: np.ndarray, hz: np.ndarray) -> str:
    """
    保存NumPy格式时间序列
    
    Parameters:
        file_path: 输出文件路径
        ex, ey, hx, hy, hz: 电磁场时间序列
    
    Returns:
        保存的文件路径
    """
    data = np.column_stack([ex, ey, hx, hy, hz])
    np.save(file_path, data)
    
    return file_path


def load_numpy_timeseries(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载NumPy格式时间序列
    
    Parameters:
        file_path: 文件路径
    
    Returns:
        (ex, ey, hx, hy, hz) 元组
    """
    data = np.load(file_path)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]


class SystemCalibrator:
    """
    系统标定器 - 调用SysCal V7进行仪器响应标定
    
    用于Phoenix MTU-5A系统的现场标定
    """
    
    def __init__(self, syscal_exe_path: str = './SysCal_V7.exe',
                 tmp_dir: str = './tmp'):
        self.syscal_exe_path = syscal_exe_path
        self.tmp_dir = tmp_dir
        self.responses: Optional[np.ndarray] = None
    
    def run_calibration(self, tbl_config, 
                        config_dir: str = './Config') -> bool:
        """
        运行SysCal V7标定程序
        
        Parameters:
            tbl_config: TBL配置文件
            config_dir: 配置文件目录
        
        Returns:
            是否成功
        """
        import subprocess
        from pathlib import Path
        
        tmp_path = Path(self.tmp_dir)
        tmp_path.mkdir(parents=True, exist_ok=True)
        
        try:
            result = subprocess.run(
                [self.syscal_exe_path],
                cwd=config_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            return result.returncode == 0
        except Exception:
            return False
    
    def get_system_responses(self, frequencies: np.ndarray,
                            channel_ex: int, channel_ey: int,
                            channel_hx: int, channel_hy: int,
                            channel_hz: int) -> np.ndarray:
        """
        获取系统响应
        
        Parameters:
            frequencies: 频率数组
            channel_*: 各通道索引
        
        Returns:
            系统响应数组 (5 x n_frequencies)
        """
        n = len(frequencies)
        responses = np.ones((5, n), dtype=complex)
        
        responses[0] = 1e-6
        responses[1] = 1e-6
        responses[2] = 1e-9
        responses[3] = 1e-9
        responses[4] = 1e-9
        
        return responses
    
    def apply_calibration(self, data: np.ndarray, 
                         channel: int, frequencies: np.ndarray) -> np.ndarray:
        """
        应用系统标定到时间序列
        
        Parameters:
            data: 时间序列数据
            channel: 通道号 (0-4)
            frequencies: 频率数组
        
        Returns:
            标定后的时间序列
        """
        if self.responses is None:
            return data
        
        fft_data = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data))
        
        for i, f in enumerate(freqs):
            if f > 0 and i < len(self.responses[channel]):
                resp = self.responses[channel, i]
                if abs(resp) > 1e-10:
                    fft_data[i] /= resp
        
        return np.fft.irfft(fft_data, n=len(data))


if __name__ == '__main__':
    site = create_test_site()
    schema = SyntheticSchema.from_ts('TS3')
    
    synth = SyntheticTimeSeries(schema, SyntheticMethod.RANDOM_SEG_PARTIAL)
    
    t1 = datetime(2023, 1, 1, 0, 0, 0)
    t2 = datetime(2023, 1, 1, 0, 0, 10)
    
    ex, ey, hx, hy, hz = synth.generate(t1, t2, site, seed=42)
    
    print(f"Generated {len(ex)} samples")
    print(f"Ex range: [{ex.min():.4f}, {ex.max():.4f}]")
    print(f"Hx range: [{hx.min():.6f}, {hx.max():.6f}]")
