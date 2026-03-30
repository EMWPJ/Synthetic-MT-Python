"""
MT工作流配置模块

定义采集系统配置和常量
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict


# 物理常数
MU0 = 4 * np.pi * 1e-7  # 真空磁导率 (H/m)


@dataclass
class TSConfig:
    """采集系统配置"""

    name: str
    sample_rate: float  # Hz
    freq_min: float  # Hz
    freq_max: float  # Hz
    period_min: float  # s
    period_max: float  # s
    description: str = ""

    def __repr__(self) -> str:
        return f"TSConfig('{self.name}', {self.sample_rate}Hz, T:[{self.period_min:.2f},{self.period_max:.2f}]s)"


# Phoenix兼容的采集系统配置
TS_CONFIGS = {
    "TS2": TSConfig(
        name="TS2",
        sample_rate=2400,
        freq_min=10,
        freq_max=12000,
        period_min=1 / 12000,
        period_max=1 / 10,
        description="High frequency 2400Hz system",
    ),
    "TS3": TSConfig(
        name="TS3",
        sample_rate=2400,
        freq_min=1,
        freq_max=1000,
        period_min=1 / 1000,
        period_max=1 / 1,
        description="Broadband 2400Hz system (1-1000Hz)",
    ),
    "TS4": TSConfig(
        name="TS4",
        sample_rate=150,
        freq_min=0.1,
        freq_max=10,
        period_min=0.1,  # 10s
        period_max=10,  # 100s
        description="Short period 150Hz system",
    ),
    "TS5": TSConfig(
        name="TS5",
        sample_rate=15,
        freq_min=1e-6,
        freq_max=1,
        period_min=1,
        period_max=1e6,  # 1e6 seconds = ~11.6 days
        description="Long period 15Hz system",
    ),
}


def get_config(name: str) -> TSConfig:
    """获取采集系统配置"""
    return TS_CONFIGS.get(name, TS_CONFIGS["TS3"])


def get_periods_for_config(config: TSConfig, n_periods: int = 16) -> Tuple[float, ...]:
    """获取配置对应的周期数组（对数等分）"""
    import numpy as np

    return tuple(
        np.logspace(np.log10(config.period_min), np.log10(config.period_max), n_periods)
    )


# 多频段联合处理配置
MULTI_BAND_CONFIG = {
    "TS3": {
        "period_range": (0.001, 1.0),  # 1ms - 1s
        "sample_rate": 2400,
        "duration": 10,  # seconds
        "n_periods": 16,
    },
    "TS4": {
        "period_range": (0.1, 10.0),  # 0.1s - 10s
        "sample_rate": 150,
        "duration": 60,  # seconds (need more for long periods)
        "n_periods": 12,
    },
    "TS5": {
        "period_range": (1.0, 1000.0),  # 1s - 1000s
        "sample_rate": 15,
        "duration": 3600,  # 1 hour for long periods
        "n_periods": 8,
    },
}


# ============================================================================
# 默认正演合成频率配置
# ============================================================================
# 正演合成使用的频率范围
# 频率: 1000Hz ~ 100000s (周期)
# 对数等间隔，每个数量级20个频点，总共161个频点
FORWARD_PERIODS_CONFIG = {
    "freq_min": 1 / 1000,  # 0.001s = 1ms (对应1000Hz)
    "freq_max": 100000,  # 100000s
    "points_per_decade": 20,  # 每个数量级20个频点
}


# 计算默认正演周期数组
def get_default_forward_periods() -> np.ndarray:
    """获取默认正演周期数组"""
    import numpy as np

    config = FORWARD_PERIODS_CONFIG
    # 从 10^-3 到 10^5，共8个数量级
    # 每个数量级20个点
    decades = np.log10(config["freq_max"]) - np.log10(config["freq_min"])  # 8 decades
    n_points = int(decades * config["points_per_decade"]) + 1  # 161 points
    periods = np.logspace(
        np.log10(config["freq_min"]), np.log10(config["freq_max"]), n_points
    )
    return periods


# ============================================================================
# 默认数据处理频点配置
# ============================================================================
# 数据处理使用的频率范围
# 周期: 300Hz ~ 20000s
# 对数等间隔，每个倍频(octave) 4个频点
PROCESSING_PERIODS_CONFIG = {
    "period_min": 1 / 300,  # 0.0033s (对应300Hz)
    "period_max": 20000,  # 20000s
    "octaves": 4,  # 每个倍频4个频点
}


# 计算默认处理周期数组
def get_default_processing_periods() -> np.ndarray:
    """获取默认处理周期数组"""
    import numpy as np

    config = PROCESSING_PERIODS_CONFIG
    # 从 period_min 到 period_max 的倍频数
    # log2(period_max / period_min) = log2(20000 / (1/300)) = log2(6e6) ≈ 22.5 octaves
    # 每个倍频4个点
    n_octaves = np.log2(config["period_max"] / config["period_min"])  # ≈ 22.5 octaves
    n_points = int(n_octaves * config["octaves"]) + 1
    periods = np.logspace(
        np.log10(config["period_min"]), np.log10(config["period_max"]), n_points
    )
    return periods


# ============================================================================
# 分段交替采集配置
# ============================================================================

BAND_MAP = {
    "HIGH": "TS3",  # 2400Hz, 1-1000Hz
    "MED": "TS4",  # 150Hz, 0.1-10Hz
    "LOW": "TS5",  # 15Hz, 1e-6-1Hz
}


@dataclass
class SegmentedAcquisitionConfig:
    """分段交替采集配置"""

    interval: int = 300  # 采集间隔(秒) - 高频/中频交替周期的一半
    high_duration: int = 2  # 高频采集时长(秒)
    med_duration: int = 16  # 中频采集时长(秒)
    total_duration: float = 600.0  # 总采集时长(秒)

    def __post_init__(self):
        """验证配置参数"""
        if self.interval <= 0:
            raise ValueError("interval must be positive")
        if self.high_duration <= 0:
            raise ValueError("high_duration must be positive")
        if self.med_duration <= 0:
            raise ValueError("med_duration must be positive")
        if self.total_duration <= 0:
            raise ValueError("total_duration must be positive")

    @property
    def cycle_duration(self) -> int:
        """完整周期时长 = interval * 2"""
        return self.interval * 2

    def generate_schedule(self) -> List[Dict]:
        """
        生成采集时间表

        Returns:
            List of dicts with keys: band, start, end, duration
            例如: [{"band": "HIGH", "start": 0.0, "end": 2.0, "duration": 2.0}, ...]
        """
        schedule = []
        current_time = 0.0

        while current_time < self.total_duration:
            # HIGH频段采集
            high_end = current_time + self.high_duration
            if high_end > self.total_duration:
                high_end = self.total_duration
            schedule.append(
                {
                    "band": "HIGH",
                    "start": current_time,
                    "end": high_end,
                    "duration": high_end - current_time,
                }
            )
            current_time += self.interval

            if current_time >= self.total_duration:
                break

            # MED频段采集
            med_end = current_time + self.med_duration
            if med_end > self.total_duration:
                med_end = self.total_duration
            schedule.append(
                {
                    "band": "MED",
                    "start": current_time,
                    "end": med_end,
                    "duration": med_end - current_time,
                }
            )
            current_time += self.interval

        return schedule
