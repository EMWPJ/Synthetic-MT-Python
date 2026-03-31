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
    SegmentedTimeSeriesSynthesizer,
    SegmentedTimeSeriesProcessor,
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

        # 测点管理器
        self.station_manager: "StationManager" = StationManager()

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

    def get_forward_results(self) -> Optional[Dict]:
        """获取正演计算结果"""
        if self.current_periods is None:
            return None
        return {
            "periods": self.current_periods,
            "app_resistivity": self.current_rho,
            "phase": self.current_phase,
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
        from synthetic_mt import ForwardSite

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

    def synthesize_time_series_for_station(
        self,
        station_name: str,
        band: str = "TS3",
        duration: float = 10.0,
        seed: Optional[int] = None,
        start_time: Optional[datetime] = None,
    ) -> Dict:
        """
        合成指定测点的时间序列

        使用测点管理器中存储的测点数据进行合成。

        Args:
            station_name: 测点名称
            band: 频段 ('TS3', 'TS4', 'TS5')
            duration: 时长 (秒)
            seed: 随机种子. 如果为None,则从start_time自动派生
            start_time: 合成开始时间. 如果为None,则使用默认值.

        Returns:
            时间序列数据字典
        """
        from synthetic_mt import ForwardSite
        from synthetic_mt.domain.entities import EMFields

        # 获取测点数据
        station = self.station_manager.get_station(station_name)
        if station is None:
            raise ValueError(f"Station not found: {station_name}")

        config = TS_CONFIGS.get(band)
        if config is None:
            raise ValueError(f"Unknown band: {band}")

        # 获取测点的周期数据
        periods = station["periods"]
        zxy = station.get("zxy")
        zyx = station.get("zyx")
        zxx = station.get("zxx")
        zyy = station.get("zyy")

        # 重建EMFields列表
        fields = []
        for i, T in enumerate(periods):
            freq = 1.0 / T
            field = EMFields(
                freq=freq,
                zxx=zxx[i] if zxx is not None else complex(0, 0),
                zxy=zxy[i] if zxy is not None else complex(0, 0),
                zyx=zyx[i] if zyx is not None else complex(0, 0),
                zyy=zyy[i] if zyy is not None else complex(0, 0),
            )
            fields.append(field)

        # 创建ForwardSite
        site = ForwardSite(
            name=station_name, x=station["x"], y=station["y"], fields=fields
        )

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
        synthetic_periods: float = 200.0,
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

    def synthesize_time_series_segmented(
        self,
        acquisition_config,  # SegmentedAcquisitionConfig
        seed: Optional[int] = None,
        start_time: Optional[datetime] = None,
    ) -> Dict:
        """
        生成分段交替采集的时间序列

        Args:
            acquisition_config: 分段采集配置 (SegmentedAcquisitionConfig)
            seed: 随机种子
            start_time: 开始时间

        Returns:
            {
                "segments": [...],  # HIGH/MED分段数据
                "low": {...},       # LOW连续数据
                "schedule": {...},  # 采集配置副本
                "seed": int,
            }
        """
        # Local import to avoid circular/relative import issues
        from ..config import SegmentedAcquisitionConfig

        # Validate station exists (need fields from forward)
        if self.forward_calc is None:
            raise RuntimeError("No model created. Run forward first.")

        # Get fields from current forward result
        if not self.current_fields:
            raise RuntimeError("No fields. Run forward first.")

        # Determine seed
        if seed is None:
            if start_time is None:
                start_time = datetime(2023, 1, 1, 0, 0, 0)
            start_ts = (start_time - datetime(1970, 1, 1)).total_seconds()
            seed = int(start_ts)

        # Create synthesizer and generate
        synth = SegmentedTimeSeriesSynthesizer(acquisition_config)
        result = synth.generate(self.current_fields, seed=seed)

        # Store result
        self.current_time_series = result

        return result

    def process_segmented_time_series(
        self,
        periods: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        处理分段采集的时间序列

        Args:
            periods: 目标周期数组 (默认使用get_default_processing_periods())

        Returns:
            {
                "periods": np.ndarray,
                "Zxx", "Zxy", "Zyx", "Zyy": 阻抗张量
                "app_resistivity": 视电阻率
                "phase": 相位
                ...
            }
        """
        if self.current_time_series is None:
            raise RuntimeError(
                "No time series. Call synthesize_time_series_segmented first."
            )

        if "segments" not in self.current_time_series:
            raise RuntimeError(
                "Time series does not contain segments. Use synthesize_time_series_segmented."
            )

        if periods is None:
            periods = get_default_processing_periods()

        segments = self.current_time_series["segments"]

        # Get sample rate from first segment
        sample_rate = segments[0].get("sample_rate") if segments else None
        if sample_rate is None:
            raise ValueError("No sample_rate found in segments")

        processor = SegmentedTimeSeriesProcessor(segments, sample_rate=sample_rate)
        result = processor.estimate_impedance_at_periods(periods)

        self.processed_results = result

        return result

    def _reconstruct_fields_from_impedance(
        self, periods: np.ndarray, impedance: Dict
    ) -> List:
        """从阻抗数据重建EMFields列表"""
        from synthetic_mt.domain.entities import EMFields

        fields = []
        for i, T in enumerate(periods):
            freq = 1.0 / T
            field = EMFields(
                freq=freq,
                zxx=impedance["Zxx"][i],
                zxy=impedance["Zxy"][i],
                zyx=impedance["Zyx"][i],
                zyy=impedance["Zyy"][i],
            )
            fields.append(field)
        return fields

    def batch_synthesize(
        self,
        station_names: List[str],
        band: str = "TS3",
        duration: float = 10.0,
        seed: Optional[int] = None,
        start_time: Optional[datetime] = None,
        synthetic_periods: float = 200.0,
        progress_callback=None,  # callable(current, total, message)
    ) -> Dict[str, Dict]:
        """
        批量合成多个测点的时间序列

        所有测点使用相同的start_time确保天然场源变化一致,
        每个测点使用不同的seed (seed + i) 避免完全相同的时间序列

        Args:
            station_names: 测点名称列表
            band: 频段 ('TS3', 'TS4', 'TS5')
            duration: 时长 (秒)
            seed: 随机种子. 如果为None,则从start_time自动派生
            start_time: 合成开始时间. 如果为None,则使用默认值.
            synthetic_periods: 合成周期数（每个频点包含的周期数，影响分段长度）
            progress_callback: 进度回调函数 (current, total, message)

        Returns:
            字典: station_name -> synthesis result
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

        # 提取阻抗数据用于重建fields
        impedance = self.forward_calc.calculate_impedance(band_periods)

        # 确定开始时间和种子
        if start_time is None:
            start_time = datetime(2023, 1, 1, 0, 0, 0)

        start_time_offset = (start_time - datetime(1970, 1, 1)).total_seconds()

        # 如果未指定种子,则从开始时间派生
        if seed is None:
            seed = int(start_time_offset) + int(duration * 1000)

        t1 = start_time
        t2 = t1 + timedelta(seconds=int(duration))

        # 批量合成
        results = {}
        total = len(station_names)

        for i, station_name in enumerate(station_names):
            # 每个测点使用不同的seed (seed + i)
            station_seed = seed + i

            # 重建fields (各测点阻抗相同,但随机序列不同)
            station_fields = self._reconstruct_fields_from_impedance(
                band_periods, impedance
            )

            # 随机分段合成
            synth = RandomSegmentTimeSeriesSynthesizer(
                sample_rate=config.sample_rate, synthetic_periods=synthetic_periods
            )
            ts_result = synth.generate_from_fields(
                station_fields,
                duration=duration,
                seed=station_seed,
                start_time=start_time_offset,
            )

            # 提取频率信息
            frequencies = np.array([f.freq for f in station_fields])

            results[station_name] = {
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
                "seed": station_seed,
            }

            # 确保测点存在于管理器中(使用合成频段的正演结果)
            if self.station_manager.get_station(station_name) is None:
                # 从阻抗数据提取复阻抗和计算视电阻率/相位
                zxx = impedance["Zxx"]
                zxy = impedance["Zxy"]
                zyx = impedance["Zyx"]
                zyy = impedance["Zyy"]

                # 计算band_periods对应的视电阻率和相位
                abs_zxy = np.abs(zxy)
                rho_a_band = (abs_zxy**2) / (MU0 * 2 * np.pi * (1.0 / band_periods))
                phase_band = np.angle(zxy, deg=True)

                self.add_station_with_data(
                    name=station_name,
                    x=0.0,  # 默认坐标
                    y=0.0,
                    periods=band_periods,
                    rho_a=rho_a_band,
                    phase=phase_band,
                    zxx=zxx,
                    zxy=zxy,
                    zyx=zyx,
                    zyy=zyy,
                )

            # 存储时间序列到测点管理器
            self.station_manager.set_time_series(station_name, results[station_name])

            # 进度回调
            if progress_callback is not None:
                progress_callback(i + 1, total, f"Processed {station_name}")

        return results

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
    # 测点管理
    # ========================================================================

    def add_station(
        self,
        name: str,
        x: float = 0.0,
        y: float = 0.0,
    ) -> Dict:
        """
        添加测点到管理器

        将当前正演结果作为测点存储。

        Args:
            name: 测点名称
            x: X坐标
            y: Y坐标

        Returns:
            添加的测点数据字典
        """
        if self.current_periods is None:
            raise RuntimeError("No forward results. Run forward first.")

        # Extract impedance from EMFields list
        zxx = None
        zxy = None
        zyx = None
        zyy = None
        if self.current_fields:
            zxx = np.array([f.zxx for f in self.current_fields])
            zxy = np.array([f.zxy for f in self.current_fields])
            zyx = np.array([f.zyx for f in self.current_fields])
            zyy = np.array([f.zyy for f in self.current_fields])

        return self.station_manager.add_station(
            name=name,
            x=x,
            y=y,
            periods=self.current_periods,
            rho_a=self.current_rho,
            phase=self.current_phase,
            zxx=zxx,
            zxy=zxy,
            zyx=zyx,
            zyy=zyy,
        )

    def add_station_with_data(
        self,
        name: str,
        x: float,
        y: float,
        periods: np.ndarray,
        rho_a: np.ndarray,
        phase: np.ndarray,
        zxx: Optional[np.ndarray] = None,
        zxy: Optional[np.ndarray] = None,
        zyx: Optional[np.ndarray] = None,
        zyy: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        添加测点（直接指定数据，不依赖当前正演结果）

        用于从XMLSite导入或在外部计算好数据后添加测点。

        Args:
            name: 测点名称
            x: X坐标
            y: Y坐标
            periods: 周期数组
            rho_a: 视电阻率数组
            phase: 相位数组
            zxx, zxy, zyx, zyy: 阻抗分量（可选）

        Returns:
            添加的测点数据字典
        """
        return self.station_manager.add_station(
            name=name,
            x=x,
            y=y,
            periods=periods,
            rho_a=rho_a,
            phase=phase,
            zxx=zxx,
            zxy=zxy,
            zyx=zyx,
            zyy=zyy,
        )

    def get_station(self, name: str) -> Optional[Dict]:
        """获取测点数据"""
        return self.station_manager.get_station(name)

    def list_stations(self) -> List[str]:
        """列出所有测点"""
        return self.station_manager.list_stations()

    def remove_station(self, name: str) -> bool:
        """移除测点"""
        return self.station_manager.remove_station(name)

    def get_time_series(self, name: str) -> Optional[Dict]:
        """
        获取测点的时间序列数据

        Args:
            name: 测点名称

        Returns:
            时间序列数据字典，包含 ex, ey, hx, hy, hz 等,如果没有则返回None
        """
        return self.station_manager.get_time_series(name)

    # ========================================================================
    # 数据导出
    # ========================================================================

    def export_station_csv(self, name: str, filepath: str) -> None:
        """
        导出单站数据到CSV文件

        Args:
            name: 站点名称
            filepath: 输出文件路径
        """
        import csv

        station = self.station_manager.get_station(name)
        if station is None:
            raise ValueError(f"Station not found: {name}")

        periods = station["periods"]
        rho_a = station["rho_a"]
        phase = station["phase"]
        zxx = station.get("zxx")
        zxy = station.get("zxy")
        zyx = station.get("zyx")
        zyy = station.get("zyy")
        has_impedance = zxx is not None or zxy is not None

        with open(filepath, "w", newline="") as f:
            if has_impedance:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Period",
                        "App_Resistivity",
                        "Phase",
                        "Zxx_Re",
                        "Zxx_Im",
                        "Zxy_Re",
                        "Zxy_Im",
                        "Zyx_Re",
                        "Zyx_Im",
                        "Zyy_Re",
                        "Zyy_Im",
                    ]
                )
                for i in range(len(periods)):
                    writer.writerow(
                        [
                            periods[i],
                            rho_a[i],
                            phase[i],
                            zxx[i].real if zxx is not None and i < len(zxx) else 0,
                            zxx[i].imag if zxx is not None and i < len(zxx) else 0,
                            zxy[i].real if zxy is not None and i < len(zxy) else 0,
                            zxy[i].imag if zxy is not None and i < len(zxy) else 0,
                            zyx[i].real if zyx is not None and i < len(zyx) else 0,
                            zyx[i].imag if zyx is not None and i < len(zyx) else 0,
                            zyy[i].real if zyy is not None and i < len(zyy) else 0,
                            zyy[i].imag if zyy is not None and i < len(zyy) else 0,
                        ]
                    )
            else:
                writer = csv.writer(f)
                writer.writerow(["Period", "App_Resistivity", "Phase"])
                for i in range(len(periods)):
                    writer.writerow([periods[i], rho_a[i], phase[i]])

    def export_station_numpy(self, name: str, filepath: str) -> None:
        """
        导出单站数据到NumPy .npz文件

        Args:
            name: 站点名称
            filepath: 输出文件路径
        """
        station = self.station_manager.get_station(name)
        if station is None:
            raise ValueError(f"Station not found: {name}")

        periods = station["periods"]
        rho_a = station["rho_a"]
        phase = station["phase"]
        x = station["x"]
        y = station["y"]
        zxx = station.get("zxx")
        zxy = station.get("zxy")
        zyx = station.get("zyx")
        zyy = station.get("zyy")

        all_data = {
            "name": name,
            "x": x,
            "y": y,
            "periods": periods,
            "app_resistivity": rho_a,
            "phase": phase,
        }

        if zxx is not None or zxy is not None:
            all_data.update(
                {
                    "zxx": zxx if zxx is not None else np.array([]),
                    "zxy": zxy if zxy is not None else np.array([]),
                    "zyx": zyx if zyx is not None else np.array([]),
                    "zyy": zyy if zyy is not None else np.array([]),
                }
            )

        np.savez(filepath, **all_data)

    def export_batch_numpy(self, station_names: List[str], filepath: str) -> None:
        """
        批量导出多站数据到单个NumPy .npz文件

        Args:
            station_names: 站点名称列表
            filepath: 输出文件路径
        """
        if len(station_names) == 0:
            raise ValueError("station_names cannot be empty")

        all_data: Dict[str, Any] = {"names": station_names}

        for station_name in station_names:
            station = self.station_manager.get_station(station_name)
            if station is None:
                raise ValueError(f"Station not found: {station_name}")

            all_data[station_name] = {
                "periods": station["periods"],
                "app_resistivity": station["rho_a"],
                "phase": station["phase"],
                "x": station["x"],
                "y": station["y"],
            }

            zxx = station.get("zxx")
            zxy = station.get("zxy")
            zyx = station.get("zyx")
            zyy = station.get("zyy")
            if zxx is not None or zxy is not None:
                all_data[station_name].update(
                    {
                        "zxx": zxx if zxx is not None else np.array([]),
                        "zxy": zxy if zxy is not None else np.array([]),
                        "zyx": zyx if zyx is not None else np.array([]),
                        "zyy": zyy if zyy is not None else np.array([]),
                    }
                )

        np.savez(filepath, **all_data)

    # ========================================================================
    # 项目管理
    # ========================================================================

    def save_project(self, filepath: str) -> None:
        """
        保存项目到JSON文件

        保存内容包括: 当前模型、测点列表、正演结果等

        Args:
            filepath: 项目文件路径 (.json)
        """
        import json

        # 准备测点数据
        stations_data = []
        for name in self.station_manager.list_stations():
            station = self.station_manager.get_station(name)
            if station is None:
                continue

            # 转换复杂数为可序列化格式
            def complex_to_list(arr):
                if arr is None:
                    return None
                return [[c.real, c.imag] for c in arr]

            station_data = {
                "name": station["name"],
                "x": float(station["x"]),
                "y": float(station["y"]),
                "periods": station["periods"].tolist(),
                "rho_a": station["rho_a"].tolist(),
                "phase": station["phase"].tolist(),
                "zxx": complex_to_list(station.get("zxx")),
                "zxy": complex_to_list(station.get("zxy")),
                "zyx": complex_to_list(station.get("zyx")),
                "zyy": complex_to_list(station.get("zyy")),
            }

            # 保存时间序列数据(如果存在)
            time_series = station.get("time_series")
            if time_series is not None:
                station_data["time_series"] = {
                    "band": time_series.get("band"),
                    "start_time": (
                        str(time_series["start_time"])
                        if time_series.get("start_time") is not None
                        else None
                    ),
                    "end_time": (
                        str(time_series["end_time"])
                        if time_series.get("end_time") is not None
                        else None
                    ),
                    "duration": float(time_series.get("duration", 0)),
                    "n_samples": int(time_series.get("n_samples", 0)),
                    "sample_rate": float(time_series.get("sample_rate", 0)),
                    "method": time_series.get("method"),
                    "seed": int(time_series.get("seed", 0)),
                    "ex": time_series["ex"].tolist()
                    if time_series.get("ex") is not None
                    else None,
                    "ey": time_series["ey"].tolist()
                    if time_series.get("ey") is not None
                    else None,
                    "hx": time_series["hx"].tolist()
                    if time_series.get("hx") is not None
                    else None,
                    "hy": time_series["hy"].tolist()
                    if time_series.get("hy") is not None
                    else None,
                    "hz": time_series["hz"].tolist()
                    if time_series.get("hz") is not None
                    else None,
                    "frequencies": (
                        time_series["frequencies"].tolist()
                        if time_series.get("frequencies") is not None
                        else None
                    ),
                }

            stations_data.append(station_data)

        # 准备模型数据
        model_data = None
        if self.model is not None:
            model_data = {
                "name": self.model.name,
                "n_layers": self.model.n_layers,
                "resistivity": self.model.resistivity.tolist(),
                "thickness": (
                    self.model.thickness.tolist()
                    if self.model.thickness is not None
                    else None
                ),
            }

        # 构建项目数据
        project = {
            "version": "1.0",
            "timestamp": np.datetime64("now").astype(str),
            "model": model_data,
            "stations": stations_data,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(project, f, indent=2, ensure_ascii=False)

    def load_project(self, filepath: str) -> None:
        """
        从JSON文件加载项目

        Args:
            filepath: 项目文件路径 (.json)
        """
        import json

        with open(filepath, "r", encoding="utf-8") as f:
            project = json.load(f)

        version = project.get("version", "1.0")

        # 加载模型
        model_data = project.get("model")
        if model_data is not None:
            resistivity = np.array(model_data["resistivity"])
            thickness = (
                np.array(model_data["thickness"])
                if model_data["thickness"] is not None
                else None
            )
            self.model = MT1DModel(
                model_data.get("name", "loaded_model"), resistivity, thickness
            )
            self.forward_calc = MT1DForward(self.model)

        # 加载测点
        stations_data = project.get("stations", [])
        self.station_manager.clear()

        def list_to_complex(arr):
            if arr is None:
                return None
            return np.array([complex(r, i) for r, i in arr])

        for station_data in stations_data:
            self.station_manager.add_station(
                name=station_data["name"],
                x=float(station_data["x"]),
                y=float(station_data["y"]),
                periods=np.array(station_data["periods"]),
                rho_a=np.array(station_data["rho_a"]),
                phase=np.array(station_data["phase"]),
                zxx=list_to_complex(station_data.get("zxx")),
                zxy=list_to_complex(station_data.get("zxy")),
                zyx=list_to_complex(station_data.get("zyx")),
                zyy=list_to_complex(station_data.get("zyy")),
            )

            # 恢复时间序列数据(如果存在)
            time_series_data = station_data.get("time_series")
            if time_series_data is not None:
                from datetime import datetime

                time_series = {
                    "band": time_series_data.get("band"),
                    "start_time": (
                        datetime.fromisoformat(time_series_data["start_time"])
                        if time_series_data.get("start_time")
                        else None
                    ),
                    "end_time": (
                        datetime.fromisoformat(time_series_data["end_time"])
                        if time_series_data.get("end_time")
                        else None
                    ),
                    "duration": float(time_series_data.get("duration", 0)),
                    "n_samples": int(time_series_data.get("n_samples", 0)),
                    "sample_rate": float(time_series_data.get("sample_rate", 0)),
                    "method": time_series_data.get("method"),
                    "seed": int(time_series_data.get("seed", 0)),
                    "ex": np.array(time_series_data["ex"])
                    if time_series_data.get("ex") is not None
                    else None,
                    "ey": np.array(time_series_data["ey"])
                    if time_series_data.get("ey") is not None
                    else None,
                    "hx": np.array(time_series_data["hx"])
                    if time_series_data.get("hx") is not None
                    else None,
                    "hy": np.array(time_series_data["hy"])
                    if time_series_data.get("hy") is not None
                    else None,
                    "hz": np.array(time_series_data["hz"])
                    if time_series_data.get("hz") is not None
                    else None,
                    "frequencies": (
                        np.array(time_series_data["frequencies"])
                        if time_series_data.get("frequencies") is not None
                        else None
                    ),
                }
                self.station_manager.set_time_series(station_data["name"], time_series)

        # 恢复current_fields(从加载的模型重新计算)
        if self.forward_calc is not None and self.model is not None:
            # 获取第一个测点的周期作为当前周期
            stations_list = self.station_manager.list_stations()
            if stations_list:
                first_station = self.station_manager.get_station(stations_list[0])
                if first_station is not None:
                    periods = first_station["periods"]
                    self.current_periods = periods
                    self.current_fields = self.forward_calc.calculate_fields(periods)
                    self.current_rho = first_station["rho_a"]
                    self.current_phase = first_station["phase"]

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
# 测点管理器
# ============================================================================


class StationManager:
    """
    测点管理器 - 管理多个测点的正演结果（不存储模型）

    用于存储和管理多个测站的正演计算结果，便于批量处理和多测点对比分析。
    """

    def __init__(self):
        self._stations: Dict[str, Dict] = {}
        self._selected: Optional[str] = None

    def add_station(
        self,
        name: str,
        x: float,
        y: float,
        periods: np.ndarray,
        rho_a: np.ndarray,
        phase: np.ndarray,
        zxx: Optional[np.ndarray] = None,
        zxy: Optional[np.ndarray] = None,
        zyx: Optional[np.ndarray] = None,
        zyy: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        添加测点

        Args:
            name: 测点名称
            x: X坐标
            y: Y坐标
            periods: 周期数组
            rho_a: 视电阻率数组
            phase: 相位数组
            zxx, zxy, zyx, zyy: 阻抗分量（可选）

        Returns:
            添加的测点数据字典

        Raises:
            ValueError: 如果名称为空、已存在或数组长度不匹配
        """
        # 验证名称不为空
        if not name or not name.strip():
            raise ValueError("Station name cannot be empty")

        name = name.strip()

        # 验证不重复
        if name in self._stations:
            raise ValueError(f"Station already exists: {name}")

        # 验证数组长度一致性
        base_len = len(periods)
        if len(rho_a) != base_len:
            raise ValueError(
                f"rho_a length ({len(rho_a)}) must match periods length ({base_len})"
            )
        if len(phase) != base_len:
            raise ValueError(
                f"phase length ({len(phase)}) must match periods length ({base_len})"
            )

        # 验证阻抗数组长度(如果提供)
        for imp_name, imp_arr in [
            ("zxx", zxx),
            ("zxy", zxy),
            ("zyx", zyx),
            ("zyy", zyy),
        ]:
            if imp_arr is not None and len(imp_arr) != base_len:
                raise ValueError(
                    f"{imp_name} length ({len(imp_arr)}) must match periods length ({base_len})"
                )

        station = {
            "name": name,
            "x": x,
            "y": y,
            "periods": periods,
            "rho_a": rho_a,
            "phase": phase,
            "zxx": zxx,
            "zxy": zxy,
            "zyx": zyx,
            "zyy": zyy,
        }
        self._stations[name] = station
        if self._selected is None:
            self._selected = name
        return station

    def get_station(self, name: str) -> Optional[Dict]:
        """获取测点数据"""
        return self._stations.get(name)

    def remove_station(self, name: str) -> bool:
        """移除测点"""
        if name in self._stations:
            del self._stations[name]
            if self._selected == name:
                self._selected = next(iter(self._stations), None)
            return True
        return False

    def list_stations(self) -> List[str]:
        """列出所有测点名称"""
        return list(self._stations.keys())

    def get_selected(self) -> Optional[Dict]:
        """获取当前选中的测点"""
        if self._selected is None:
            return None
        return self._stations.get(self._selected)

    def clear(self):
        """清空所有测点"""
        self._stations.clear()
        self._selected = None

    def set_time_series(self, name: str, time_series: Dict) -> bool:
        """
        设置测点的时间序列数据

        Args:
            name: 测点名称
            time_series: 时间序列数据字典，包含 ex, ey, hx, hy, hz 等

        Returns:
            是否成功
        """
        if name not in self._stations:
            return False
        self._stations[name]["time_series"] = time_series
        return True

    def get_time_series(self, name: str) -> Optional[Dict]:
        """
        获取测点的时间序列数据

        Args:
            name: 测点名称

        Returns:
            时间序列数据字典，或None
        """
        station = self._stations.get(name)
        if station is None:
            return None
        return station.get("time_series")


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
