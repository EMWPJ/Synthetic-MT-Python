"""
后台API接口层

提供统一的API接口供GUI调用，与算法实现分离。
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

from .core import (
    MT1DModel,
    MT1DForward,
    TimeSeriesSynthesizer,
    TimeSeriesProcessor,
    Model1DValidator,
    ResultsComparator,
    TSConfig,
    TS_CONFIGS,
    get_default_forward_periods,
    get_default_processing_periods,
    MU0,
)


# ============================================================================
# API接口
# ============================================================================


class MTWorkflowAPI:
    """
    MT工作流API - 提供给GUI的统一接口

    分离算法实现与界面调用
    """

    def __init__(self):
        self.model: Optional[MT1DModel] = None
        self.forward_calc: Optional[MT1DForward] = None
        self.current_fields: List = []

        # 存储当前数据
        self.current_periods: Optional[np.ndarray] = None
        self.current_rho: Optional[np.ndarray] = None
        self.current_phase: Optional[np.ndarray] = None
        self.current_time_series: Optional[Dict] = None
        self.processed_results: Optional[Dict] = None

    # ========================================================================
    # 模型管理
    # ========================================================================

    def create_model(
        self, name: str, resistivity: List[float], thickness: List[float] = None
    ) -> MT1DModel:
        """创建1D模型"""
        self.model = MT1DModel(name, resistivity, thickness)
        self.forward_calc = MT1DForward(self.model)
        return self.model

    def create_halfspace(self, name: str, rho: float) -> MT1DModel:
        """创建均匀半空间模型"""
        return self.create_model(name, [rho])

    def create_layered(
        self, name: str, resistivity: List[float], thickness: List[float]
    ) -> MT1DModel:
        """创建层状模型"""
        if len(thickness) != len(resistivity) - 1:
            raise ValueError("厚度数量 = 层数 - 1")
        return self.create_model(name, resistivity, thickness)

    def get_preset_model(self, preset: str) -> MT1DModel:
        """获取预定义模型"""
        presets = {
            "uniform_100": ([100.0], []),
            "uniform_1000": ([1000.0], []),
            "two_layer_hl": ([1000.0, 10.0], [100.0]),
            "two_layer_ll": ([10.0, 1000.0], [100.0]),
            "three_layer_hll": ([1000.0, 10.0, 100.0], [50.0, 200.0]),
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}")
        res, thick = presets[preset]
        return self.create_model(preset, res, thick)

    # ========================================================================
    # 正演计算
    # ========================================================================

    def run_forward(self, periods: Optional[np.ndarray] = None) -> Dict:
        """运行正演计算"""
        if self.forward_calc is None:
            raise RuntimeError("No model created. Call create_model first.")

        if periods is None:
            periods = get_default_forward_periods()

        self.current_periods = periods
        impedance = self.forward_calc.calculate_impedance(periods)
        rho_a, phase = self.forward_calc.calculate_app_resistivity_phase(periods)

        self.current_rho = rho_a
        self.current_phase = phase

        # 计算EMFields用于验证
        self.current_fields = self.forward_calc.calculate_fields(periods)

        return {
            "periods": periods,
            "frequencies": 1.0 / periods,
            "impedance": impedance,
            "app_resistivity": rho_a,
            "phase": phase,
            "zxy": impedance["Zxy"],
            "zyx": impedance["Zyx"],
        }

    # ========================================================================
    # 时间序列合成
    # ========================================================================

    def synthesize_time_series(
        self,
        band: str = "TS3",
        duration: float = 10.0,
        seed: Optional[int] = None,
        start_time: Optional[datetime] = None,
    ) -> Dict:
        """
        合成时间序列

        重要: 随机种子与时间绑定 - 确保相同时段产生相同的合成结果
        """
        from synthetic_mt import ForwardSite, nature_magnetic_amplitude

        if self.forward_calc is None:
            raise RuntimeError("No model created. Run forward first.")

        config = TS_CONFIGS.get(band)
        if config is None:
            raise ValueError(f"Unknown band: {band}")

        # 计算用于合成的频段周期
        band_periods = np.logspace(
            np.log10(config.period_min), np.log10(config.period_max), 16
        )

        # 创建ForwardSite
        fields = self.forward_calc.calculate_fields(band_periods)
        site = ForwardSite(name=f"Site_{self.model.name}", x=0.0, y=0.0, fields=fields)

        # 确定开始时间
        if start_time is None:
            start_time = datetime(2023, 1, 1, 0, 0, 0)

        # 如果未指定种子,则从开始时间派生
        if seed is None:
            start_ts = (start_time - datetime(1970, 1, 1)).total_seconds()
            seed = int(start_ts) + int(duration * 1000)

        t1 = start_time
        t2 = t1 + timedelta(seconds=int(duration))

        # 合成时间序列
        synth = TimeSeriesSynthesizer(config)
        ex, ey, hx, hy, hz = synth.generate(site, t1, t2, seed=seed)

        self.current_time_series = {
            "band": band,
            "config": config,
            "ex": ex,
            "ey": ey,
            "hx": hx,
            "hy": hy,
            "hz": hz,
            "start_time": t1,
            "end_time": t2,
            "duration": duration,
            "n_samples": len(ex),
            "sample_rate": config.sample_rate,
            "seed": seed,
        }

        return self.current_time_series

    def synthesize_time_series_random(
        self,
        band: str = "TS3",
        duration: float = 10.0,
        seed: Optional[int] = None,
        synthetic_periods: float = 8.0,
        start_time: Optional[datetime] = None,
    ) -> Dict:
        """
        随机分段合成时间序列 - 论文RANDOM_SEG_PARTIAL算法

        使用与论文一致的随机振幅/相位扰动，模拟真实自然源变化。
        处理结果不会精确匹配正演（这是预期行为）。

        重要: 随机种子与时间绑定
        - 如果不指定seed, 则自动从start_time派生种子,确保相同时段产生相同的随机序列
        - 这保证了多个测点在同一时段记录时,天然场源的随机扰动完全一致
        - 这符合天然场源对所有测点同时、同步、相同的物理特性

        Args:
            band: 频段 ('TS3', 'TS4', 'TS5')
            duration: 时长 (秒)
            seed: 随机种子. 如果为None,则从start_time自动派生
            synthetic_periods: 合成周期数（每个频点包含的周期数，影响分段长度）
            start_time: 合成开始时间. 如果为None,则使用默认值.
                        用于: 1) 计算deltaPha相位偏移 2) 派生随机种子
        """
        from .core import RandomSegmentTimeSeriesSynthesizer

        if self.forward_calc is None:
            raise RuntimeError("No model created. Run forward first.")

        config = TS_CONFIGS.get(band)
        if config is None:
            raise ValueError(f"Unknown band: {band}")

        # 计算用于合成的频段周期
        band_periods = np.logspace(
            np.log10(config.period_min), np.log10(config.period_max), 16
        )

        # 计算fields (包含阻抗)
        fields = self.forward_calc.calculate_fields(band_periods)

        # 确定开始时间和种子
        # 默认开始时间 (可追溯的固定值)
        if start_time is None:
            start_time = datetime(2023, 1, 1, 0, 0, 0)

        # 计算开始时间偏移(秒),用于deltaPha计算
        start_time_offset = (start_time - datetime(1970, 1, 1)).total_seconds()

        # 如果未指定种子,则从开始时间派生
        # 物理意义: 相同的时段产生相同的天然场源随机扰动
        if seed is None:
            # 使用时间戳的整数部分作为种子,确保相同秒产生相同种子
            seed = int(start_time_offset) + int(
                duration * 1000
            )  # 加入duration防止边界情况

        t1 = start_time
        t2 = t1 + timedelta(seconds=int(duration))

        # 随机分段合成 - 传入start_time_offset用于相位计算
        synth = RandomSegmentTimeSeriesSynthesizer(
            sample_rate=config.sample_rate, synthetic_periods=synthetic_periods
        )
        ts_result = synth.generate_from_fields(
            fields,
            duration=duration,
            seed=seed,
            start_time=start_time_offset,
        )

        # 提取频率信息
        frequencies = np.array([f.freq for f in fields])

        self.current_time_series = {
            "band": band,
            "config": config,
            "ex": ts_result["ex"],
            "ey": ts_result["ey"],
            "hx": ts_result["hx"],
            "hy": ts_result["hy"],
            "hz": ts_result["hz"],
            "start_time": t1,
            "end_time": t2,
            "duration": duration,
            "n_samples": ts_result["n_samples"],
            "sample_rate": config.sample_rate,
            "frequencies": frequencies,
            "method": "random_seg_partial",
            "seed": seed,  # 记录使用的种子,便于多测点验证
        }

        return self.current_time_series

    # ========================================================================
    # 数据处理
    # ========================================================================

    def process_time_series(self, periods: Optional[np.ndarray] = None) -> Dict:
        """处理时间序列得到视电阻率和相位"""
        if self.current_time_series is None:
            raise RuntimeError("No time series. Call synthesize_time_series first.")

        if periods is None:
            periods = get_default_processing_periods()

        ts = self.current_time_series
        processor = TimeSeriesProcessor(
            ts["ex"], ts["ey"], ts["hx"], ts["hy"], ts["hz"], ts["sample_rate"]
        )

        result = processor.estimate_impedance_at_periods(periods)
        self.processed_results = result

        return result

    # ========================================================================
    # 验证
    # ========================================================================

    def validate_1d_model(self) -> List[Dict]:
        """验证1D模型特征"""
        if not self.current_fields:
            raise RuntimeError("No fields. Run forward first.")

        validator = Model1DValidator(self.current_fields)
        return validator.validate_all()

    def compare_results(
        self,
        estimated_rho: Optional[np.ndarray] = None,
        estimated_phase: Optional[np.ndarray] = None,
    ) -> ResultsComparator:
        """对比正演与处理结果"""
        if self.current_periods is None:
            raise RuntimeError("No forward results. Run forward first.")

        if estimated_rho is None and self.processed_results is not None:
            estimated_rho = self.processed_results["app_resistivity"]
            estimated_phase = self.processed_results["phase"]

        return ResultsComparator(
            self.current_periods,
            self.current_rho,
            self.current_phase,
            estimated_rho,
            estimated_phase,
        )

    # ========================================================================
    # 便捷方法
    # ========================================================================

    def run_full_workflow(
        self, model: str = "uniform_100", periods: Optional[np.ndarray] = None
    ) -> Dict:
        """运行完整工作流: 正演 -> 合成 -> 处理 -> 对比"""
        # 1. 创建模型
        self.get_preset_model(model)

        # 2. 正演计算
        forward_results = self.run_forward(periods)

        # 3. 验证1D特征
        validation = self.validate_1d_model()

        # 4. 合成时间序列
        ts_results = self.synthesize_time_series()

        # 5. 处理时间序列
        processed = self.process_time_series()

        # 6. 对比结果
        comparator = self.compare_results()
        rho_error = comparator.compute_rho_error()
        phase_error = comparator.compute_phase_error()

        return {
            "model": str(self.model),
            "forward": forward_results,
            "validation": validation,
            "time_series": {
                "band": ts_results["band"],
                "duration": ts_results["duration"],
                "sample_rate": ts_results["sample_rate"],
                "n_samples": ts_results["n_samples"],
            },
            "processed": {
                "periods": processed["periods"],
                "app_resistivity": processed["app_resistivity"],
                "phase": processed["phase"],
            },
            "comparison": {
                "rho_error": rho_error,
                "phase_error": phase_error,
            },
        }


# ============================================================================
# 单例模式 - 便于GUI调用
# ============================================================================

_api_instance: Optional[MTWorkflowAPI] = None


def get_api() -> MTWorkflowAPI:
    """获取API单例"""
    global _api_instance
    if _api_instance is None:
        _api_instance = MTWorkflowAPI()
    return _api_instance


def reset_api():
    """重置API"""
    global _api_instance
    _api_instance = None
