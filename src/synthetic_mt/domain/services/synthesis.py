"""Domain service - Magnetotelluric time series synthesis.

This module contains pure business logic for synthesizing MT time series
from forward modeling data. No external dependencies (no file I/O, no network).
"""

from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np

from ..entities import EMFields, ForwardSite, nature_magnetic_amplitude
from ..value_objects import TS_CONFIGS, SyntheticMethod


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
    from ..value_objects import SyntheticMethod  # Import here to avoid circular dependency at module level

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


class SyntheticTimeSeries:
    """合成时间序列主类"""

    def __init__(self, schema: SyntheticSchema,
                 method: Optional[SyntheticMethod] = None):
        self.schema = schema
        self.method = method if method is not None else SyntheticMethod.RANDOM_SEG_PARTIAL

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
        from ..value_objects import SyntheticMethod

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
