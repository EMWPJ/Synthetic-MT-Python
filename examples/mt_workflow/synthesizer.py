"""
时间序列合成器

支持多频段(TS3/TS4/TS5)的MT时间序列合成
"""

import numpy as np
import sys
import os
from datetime import datetime
from typing import Tuple, Optional, Dict

# 添加项目路径
_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(_file))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from synthetic_mt import (
    EMFields,
    ForwardSite,
    SyntheticTimeSeries,
    SyntheticSchema,
    SyntheticMethod,
)
from .config import TSConfig, get_config
from .model_1d import MT1DModel
from .forward_1d import MT1DForward


class TimeSeriesSynthesizer:
    """
    MT时间序列合成器

    支持多频段配置
    """

    def __init__(self, config: TSConfig, method: SyntheticMethod = None):
        self.config = config
        self.method = method or SyntheticMethod.RANDOM_SEG_PARTIAL

        # 创建SyntheticSchema
        self.schema = SyntheticSchema(
            name=config.name,
            sample_rate=config.sample_rate,
            freq_min=config.freq_min,
            freq_max=config.freq_max,
        )

        # 创建合成器
        self.synth = SyntheticTimeSeries(self.schema, self.method)

    def generate(
        self, site: ForwardSite, t1: datetime, t2: datetime, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, ...]:
        """
        生成时间序列

        Args:
            site: 正演测点
            t1: 开始时间
            t2: 结束时间
            seed: 随机种子

        Returns:
            (ex, ey, hx, hy, hz) 时间序列数组
        """
        return self.synth.generate(t1, t2, site, seed=seed)

    def generate_duration(
        self,
        site: ForwardSite,
        duration: float,
        start_time: Optional[datetime] = None,
        seed: Optional[int] = None,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, datetime, datetime
    ]:
        """
        按指定时长生成时间序列

        Args:
            site: 正演测点
            duration: 时长 (秒)
            start_time: 开始时间，默认当前时刻
            seed: 随机种子

        Returns:
            (ex, ey, hx, hy, hz, t1, t2)
        """
        if start_time is None:
            start_time = datetime(2023, 1, 1, 0, 0, 0)

        t1 = start_time
        t2 = start_time.replace(second=int(start_time.second + duration))

        ex, ey, hx, hy, hz = self.generate(site, t1, t2, seed=seed)
        return ex, ey, hx, hy, hz, t1, t2


class MultiBandSynthesizer:
    """
    多频段联合合成器

    同时支持TS3, TS4, TS5三个频段
    """

    def __init__(self, configs: Optional[Dict[str, TSConfig]] = None):
        if configs is None:
            from .config import TS_CONFIGS

            configs = TS_CONFIGS

        self.synthesizers = {
            name: TimeSeriesSynthesizer(config) for name, config in configs.items()
        }

    def generate_band(
        self, band: str, site: ForwardSite, duration: float, seed: Optional[int] = None
    ) -> Dict:
        """
        生成指定频段的时间序列

        Args:
            band: 频段名称 ('TS3', 'TS4', 'TS5')
            site: 正演测点
            duration: 时长 (秒)
            seed: 随机种子

        Returns:
            包含时间序列和元数据的字典
        """
        if band not in self.synthesizers:
            raise ValueError(
                f"Unknown band: {band}. Available: {list(self.synthesizers.keys())}"
            )

        synth = self.synthesizers[band]

        # 计算采样点数
        n_samples = int(duration * synth.schema.sample_rate)

        # 生成
        start_time = datetime(2023, 1, 1, 0, 0, 0)
        t1 = start_time
        t2 = datetime(2023, 1, 1, 0, 0, int(duration))

        ex, ey, hx, hy, hz = synth.generate(site, t1, t2, seed=seed)

        return {
            "band": band,
            "config": synth.config,
            "time_series": (ex, ey, hx, hy, hz),
            "start_time": t1,
            "end_time": t2,
            "duration": duration,
            "n_samples": len(ex),
            "sample_rate": synth.schema.sample_rate,
        }

    def generate_multi_band(
        self,
        site: ForwardSite,
        durations: Optional[Dict[str, float]] = None,
        seeds: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Dict]:
        """
        生成多个频段的时间序列

        Args:
            site: 正演测点
            durations: 各频段时长 {'TS3': 10, 'TS4': 60, 'TS5': 300}
            seeds: 各频段随机种子

        Returns:
            各频段的合成结果字典
        """
        if durations is None:
            durations = {"TS3": 10, "TS4": 60, "TS5": 300}

        if seeds is None:
            seeds = {"TS3": 42, "TS4": 43, "TS5": 44}

        results = {}
        for band, duration in durations.items():
            if band in self.synthesizers:
                results[band] = self.generate_band(
                    band, site, duration, seed=seeds.get(band)
                )

        return results


def create_site_for_periods(
    model: MT1DModel, periods: np.ndarray, forward_calc: MT1DForward
) -> ForwardSite:
    """
    根据周期数组创建测点

    Args:
        model: 1D模型
        periods: 周期数组
        forward_calc: 正演计算器

    Returns:
        ForwardSite对象
    """
    fields = forward_calc.calculate_fields(periods)
    return ForwardSite(name=f"Site_{model.name}", x=0.0, y=0.0, fields=fields)
